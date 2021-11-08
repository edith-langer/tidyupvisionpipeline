#include <object_matching.h>

Eigen::Vector3f rgb2lab(const Eigen::Vector3i &rgb) {
    cv::Mat rgb_cv (1,1, CV_8UC3);  //this has some information loss because Lab values are also just uchar and not float
    rgb_cv.at<cv::Vec3b>(0,0)[0] = rgb[0];
    rgb_cv.at<cv::Vec3b>(0,0)[1] = rgb[1];
    rgb_cv.at<cv::Vec3b>(0,0)[2] = rgb[2];
    cv::Mat lab_cv;
    cv::cvtColor(rgb_cv, lab_cv, CV_RGB2Lab);

    Eigen::Vector3f lab; //from opencv lab to orig lab definition
    lab << lab_cv.at<cv::Vec3b>(0,0)[0] / 2.55f, lab_cv.at<cv::Vec3b>(0,0)[1] - 128.f, lab_cv.at<cv::Vec3b>(0,0)[2] - 128.f;
    return lab;
}

ObjectMatching::ObjectMatching(std::vector<DetectedObject> model_vec, std::vector<DetectedObject> object_vec,
                               std::string model_path, std::string cfg_path, std::string obj_match_dir) {
    model_vec_ = model_vec;
    object_vec_ = object_vec;
    model_path_ = model_path;
    cfg_path_ = cfg_path;

    if (obj_match_dir=="") {
        boost::filesystem::path model_path_orig(model_path_);
        cloud_matches_dir_ =  model_path_orig.remove_trailing_separator().parent_path().string() + "/matches/";
    } else {
        cloud_matches_dir_ = obj_match_dir;
    }
}


std::vector<Match> ObjectMatching::compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result)
{

    // setup recognizer options
    //---------------------------------------------------------------------------


    bf::path config_file = bf::path(cfg_path_);

    int verbosity = 0;
    bool force_retrain = false;  // if true, will retrain object models even if trained data already exists

    po::options_description desc("PPF Object Instance Recognizer\n"
                                 "==============================\n"
                                 "     **Allowed options**\n");

    po::variables_map vm;

    ppf_params.init(desc);

    if (v4r::io::existsFile(config_file)) {
        std::ifstream f(config_file.string());
        po::parsed_options config_parsed = po::parse_config_file(f, desc);
        po::store(config_parsed, vm);
        f.close();
    } else {
        std::cerr << config_file.string() << " does not exist!" << std::endl;
    }

    try {
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return {};
    }

    if (verbosity >= 0) {
        FLAGS_v = verbosity;
        std::cout << "Enabling verbose logging." << std::endl;
    } else {
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }


    std::vector<ObjectHypothesesStruct> global_scene_hypotheses;

    /// setup recognizer
    auto start = std::chrono::high_resolution_clock::now();
    //omp_set_num_threads(1);
    rec_.reset(new v4r::apps::PPFRecognizer<pcl::PointXYZRGB>{ppf_params});
    rec_->setModelsDir(model_path_);
    rec_->setup(force_retrain);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(stop-start);
    std::cout << "Time to train the models " << duration.count() << " minutes" << std::endl;

    // use recognizer
    //---------------------------------------------------------------------------
    //std::vector<std::string> objects_to_look_for{};   // empty vector means look for all objects

    //call the recognizer, filter hypotheses
    global_scene_hypotheses = createHypotheses();

    /// first try to match single standing objects --> therefore used min_fitness_weight for confidence computation between two objects
    /// if we find a match, we assume that the whole cluster is one object (because the confidence has to be high for both model and object)
    /// remove hypothesis with a confidence lower than min_fitness_thr
    std::function<float(FitnessScoreStruct)> minFitness =
            [](FitnessScoreStruct s) {
        return std::min(s.model_conf, s.object_conf);
    };
    std::vector<Match> model_obj_matches_single_standing = weightedGraphMatching(global_scene_hypotheses, minFitness,  ppf_params.single_obj_min_fitness_weight_thr_);
    for (size_t m = 0; m < model_obj_matches_single_standing.size(); m++) {
        Match &match = model_obj_matches_single_standing[m];
        int obj_id = match.object_id;
        int model_id = match.model_id;
        std::vector<DetectedObject>::iterator co_iter = std::find_if( object_vec_.begin(), object_vec_.end(),[&obj_id](DetectedObject const &o) {return o.getID() == obj_id; });
        std::vector<DetectedObject>::iterator ro_iter = std::find_if( model_vec_.begin(), model_vec_.end(),[&model_id](DetectedObject const &o) {return o.getID() == model_id; });

        //compute distance between object and model
        //float dist = estimateDistance(co_iter->getObjectCloud(), ro_iter->getObjectCloud(), match.transform);
        float dist = computeMeanPointDistance(co_iter->getObjectCloud(), ro_iter->getObjectCloud());
        bool is_static = dist < max_dist_for_being_static;

        ro_iter->match_ = match;
        co_iter->match_ = match;
        if (is_static) {
            ro_iter->state_ = ObjectState::STATIC;
            co_iter->state_ = ObjectState::STATIC;
        } else {
            ro_iter->state_ = ObjectState::DISPLACED;
            co_iter->state_ = ObjectState::DISPLACED;
        }

        //delete hypothesis of object
        global_scene_hypotheses.erase(std::remove_if(global_scene_hypotheses.begin(),global_scene_hypotheses.end(), [obj_id](const auto &a){ return a.object_id == obj_id ; }), global_scene_hypotheses.end());

        //delete model hypotheses
        for (size_t g = 0; g < global_scene_hypotheses.size(); g++) {
            std::map<int, HypothesesStruct> & model_h = global_scene_hypotheses[g].model_hyp;
            if (model_h.find(model_id) != model_h.end())
                model_h.erase(model_id);
        }
    }
    //TODO: maybe delete global hypotheses with model_hyp.size == 0

    //now try to split clustered objects using bipartite graph in a loop
    //to be able to separate objects, do this as long as a result can be found
    std::map<int, pcl::PointCloud<PointNormal>::ConstPtr> model_id_cloud;
    for (size_t m = 0; m < model_vec_.size(); m++) {
        model_id_cloud.insert(std::make_pair(model_vec_[m].getID(), model_vec_[m].getObjectCloud()));
    }

    std::function<float(FitnessScoreStruct)> avgFitness =
            [](FitnessScoreStruct s) {
        if (std::min(s.object_conf, s.model_conf) < ppf_params.min_avg_fitness_weight_thr_)
            return 0.0f;
        return (s.object_conf + s.model_conf) / 2;
    };
    std::vector<Match> model_obj_matches;
    std::vector<Match> model_obj_matches_single_run = weightedGraphMatching(global_scene_hypotheses, avgFitness, ppf_params.avg_fitness_weight_thr_);
    while (model_obj_matches_single_run.size() > 0) {
        //remove already matched hypotheses
        model_obj_matches.insert(model_obj_matches.end(), model_obj_matches_single_run.begin(), model_obj_matches_single_run.end());
        for (size_t i = 0; i < model_obj_matches_single_run.size(); i++) {
            Match match = model_obj_matches_single_run[i];
            const int& obj_id = match.object_id;
            const int& model_id = match.model_id;

            std::vector<DetectedObject>::iterator co_iter = std::find_if( object_vec_.begin(), object_vec_.end(),[&obj_id](DetectedObject const &o) {return o.getID() == obj_id; });
            std::vector<DetectedObject>::iterator ro_iter = std::find_if( model_vec_.begin(), model_vec_.end(),[&model_id](DetectedObject const &o) {return o.getID() == model_id; });

            //get the original model and object clouds for the  match
            const pcl::PointCloud<PointNormal>::ConstPtr model_cloud = ro_iter->getObjectCloud();
            const pcl::PointCloud<PointNormal>::ConstPtr object_cloud = co_iter->getObjectCloud();

            pcl::PointCloud<PointNormal>::Ptr model_aligned(new pcl::PointCloud<PointNormal>());
            pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned, match.transform);

            //remove points from MODEL object input cloud
            SceneDifferencingPoints scene_diff(point_dist_diff); //this influences how many of the neighbouring points get added to the model
            std::vector<int> diff_ind;
            std::vector<int> corresponding_ind;
            SceneDifferencingPoints::Cloud::Ptr model_diff_cloud = scene_diff.computeDifference(model_aligned, object_cloud, diff_ind, corresponding_ind);

            //remove very small clusters from the diff
            std::vector<int> small_cluster_ind;
            ObjectMatching::clusterOutliersBySize(model_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);
            for (int i = small_cluster_ind.size() - 1; i >= 0; i--) {
                model_diff_cloud->points.erase(model_diff_cloud->points.begin() + small_cluster_ind[i]);
                diff_ind.erase(diff_ind.begin() + small_cluster_ind[i]);
            }
            model_diff_cloud->width = model_diff_cloud->points.size();
            //TODO add the small_Cluster_ind to the model_cloud

            //if the leftover is just a plane, too small or has little volume, remove it
            pcl::PointCloud<PointNormal>::Ptr ds_cloud = downsampleCloudVG(model_diff_cloud, ds_leaf_size_ppf);
            if (isObjectUnwanted(ds_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9))
                model_diff_cloud->clear();

            //nothing is left of the diff --> the whole model cluster is a match
            if (model_diff_cloud->size() == 0 ) { //we do not split the model object
                ro_iter->match_ = match;
                model_diff_cloud->clear();
            }
            //split the model cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_model_part(new pcl::PointCloud<PointNormal>);
                model_diff_cloud.reset(new pcl::PointCloud<PointNormal>);
                matchedPartGrowing(model_aligned, matched_model_part, model_diff_cloud, match.fitness_score.model_overlapping_pts);

//                //re-compute fitness for the splitted part
//                match.fitness_score = computeModelFitness(object_cloud, matched_model_part, ppf_params);

                //transform back to original scene
                pcl::transformPointCloudWithNormals(*matched_model_part, *matched_model_part, match.transform.inverse());
                pcl::transformPointCloudWithNormals(*model_diff_cloud, *model_diff_cloud, match.transform.inverse());

                //the matched part of the model
                ro_iter->setObjectCloud(matched_model_part);
                ro_iter->match_ = match;
                ro_iter->object_folder_path_="";

                //we add the diff_model_part later to the model_vec because it invalidates the vector
            }

            //remove points from OBJECT object input cloud
            diff_ind.clear(); corresponding_ind.clear();
            pcl::PointCloud<PointNormal>::Ptr object_diff_cloud = scene_diff.computeDifference(object_cloud, model_aligned, diff_ind, corresponding_ind);

            small_cluster_ind.clear();
            ObjectMatching::clusterOutliersBySize(object_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);
            for (int i = small_cluster_ind.size() - 1; i >= 0; i--) {
                object_diff_cloud->points.erase(object_diff_cloud->points.begin() + small_cluster_ind[i]);
                diff_ind.erase(diff_ind.begin() + small_cluster_ind[i]);
            }
            object_diff_cloud->width = object_diff_cloud->points.size();
            //TODO add the small_Cluster_ind to the object_cloud

            //if the leftover is just a plane, too small or has little volume, remove it
            ds_cloud = downsampleCloudVG(object_diff_cloud, ds_leaf_size_ppf);
            if (isObjectUnwanted(ds_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9))
                object_diff_cloud->clear();

            if (object_diff_cloud->size() == 0) { //if less than 100 points left, we do not split the object
                co_iter->match_ = match;
                object_diff_cloud->clear();
            }
            //split the object cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_object_part(new pcl::PointCloud<PointNormal>);
                object_diff_cloud.reset(new pcl::PointCloud<PointNormal>);
                matchedPartGrowing(object_cloud, matched_object_part, object_diff_cloud, match.fitness_score.object_overlapping_pts);

//                //re-compute fitness for the splitted part
//                match.fitness_score = computeModelFitness(matched_object_part, model_aligned, ppf_params);


                //the matched part of the object
                co_iter->setObjectCloud(matched_object_part);
                co_iter->match_ = match;
                co_iter->object_folder_path_="";

                //we add the diff_object_part later to the object_vec because it invalidates the vector

                //save the new partly matched object
                boost::filesystem::path model_path_orig(model_path_);
                std::string cloud_matches_dir =  cloud_matches_dir_ + "/object" + std::to_string(match.object_id) + "_part_match";
                boost::filesystem::create_directories(cloud_matches_dir);
                std::string result_cloud_path = cloud_matches_dir + "/conf_" + std::to_string(match.fitness_score.object_conf) + "_" + std::to_string(match.fitness_score.model_conf) + "_model_" + std::to_string(match.model_id) + "_" +
                        (ppf_params.ppf_rec_pipeline_.use_color_ ? "_color" : "");
                saveCloudResults(matched_object_part, model_aligned, result_cloud_path);


                //pcl::io::savePCDFile("/home/edith/object_part.pcd", *matched_object_part);
                //pcl::io::savePCDFile("/home/edith/object_remaining.pcd", *object_diff_cloud);
            }

            //recompute fitness, otherwise overlapping_point_ids do not match with the stored cloud
            pcl::transformPointCloudWithNormals(*(ro_iter->getObjectCloud()), *model_aligned, match.transform);
            FitnessScoreStruct f = computeModelFitness(co_iter->getObjectCloud(), model_aligned, ppf_params);
            ro_iter->match_.fitness_score = f;
            co_iter->match_.fitness_score = f;

            //model and/or object got split, compute distance with the matched parts
            //float dist = estimateDistance(object_cloud, model_cloud, match.transform);
            float dist = computeMeanPointDistance(co_iter->getObjectCloud(), ro_iter->getObjectCloud());
            bool is_static = dist < max_dist_for_being_static;
            ro_iter->state_ =(is_static ? ObjectState::STATIC : ObjectState::DISPLACED);
            co_iter->state_ = (is_static ? ObjectState::STATIC : ObjectState::DISPLACED);


            //ATTENTION: The ro_iter is invalid now because of the push_back. Vector re-allocates and invalidates pointers
            if (model_diff_cloud->size() > 0) {
                //if more than one diff_cloud_cluster then the ro_iter is invalid for the second cluster
                const DetectedObject ro_iter_copy = *ro_iter;
                //the remaining part of the model
                std::vector<int> small_cluster_ind;
                std::vector<pcl::PointIndices> diff_cloud_cluster_ind = ObjectMatching::clusterOutliersBySize(model_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);
                for (size_t c = 0; c < diff_cloud_cluster_ind.size(); c++) {
                    pcl::PointCloud<PointNormal>::Ptr remaining_cluster_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::ExtractIndices<PointNormal> extract;
                    extract.setInputCloud (model_diff_cloud);
                    pcl::PointIndices::Ptr cluster_ind(new pcl::PointIndices);
                    cluster_ind->indices = diff_cloud_cluster_ind[c].indices;
                    extract.setIndices (cluster_ind);
                    extract.setNegative (false);
                    extract.setKeepOrganized(false);
                    extract.filter(*remaining_cluster_cloud);

                    pcl::PointCloud<PointNormal>::Ptr ds_cloud = downsampleCloudVG(remaining_cluster_cloud, ds_leaf_size_ppf);
                    if (!isObjectUnwanted(ds_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9)) {
                        DetectedObject diff_model_part(remaining_cluster_cloud, ro_iter_copy.plane_cloud_, ro_iter_copy.plane_coeffs_, ObjectState::REMOVED, "");
                        model_vec_.push_back(diff_model_part);
                    }
                }
            }

            //ATTENTION: The co_iter is invalid now because of the push_back. Vector re-allocates and invalidates pointers
            if (object_diff_cloud->size() > 0) {
                //if more than one diff_cloud_cluster then the co_iter is invalid for the second cluster
                const DetectedObject co_iter_copy = *co_iter;
                //the remaining part of the object
                std::vector<int> small_cluster_ind;
                std::vector<pcl::PointIndices> diff_cloud_cluster_ind = ObjectMatching::clusterOutliersBySize(object_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);
                for (size_t c = 0; c < diff_cloud_cluster_ind.size(); c++) {
                    pcl::PointCloud<PointNormal>::Ptr remaining_cluster_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::ExtractIndices<PointNormal> extract;
                    extract.setInputCloud (object_diff_cloud);
                    pcl::PointIndices::Ptr cluster_ind(new pcl::PointIndices);
                    cluster_ind->indices = diff_cloud_cluster_ind[c].indices;
                    extract.setIndices (cluster_ind);
                    extract.setNegative (false);
                    extract.setKeepOrganized(false);
                    extract.filter(*remaining_cluster_cloud);

                    pcl::PointCloud<PointNormal>::Ptr ds_cloud = downsampleCloudVG(remaining_cluster_cloud, ds_leaf_size_ppf);
                    if (!isObjectUnwanted(ds_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9)) {
                        DetectedObject diff_object_part(remaining_cluster_cloud, co_iter_copy.plane_cloud_, co_iter_copy.plane_coeffs_, ObjectState::NEW, "");
                        object_vec_.push_back(diff_object_part);
                    }
                }
            }
        }

        //call recognizer again
        global_scene_hypotheses = createHypotheses();
        if (global_scene_hypotheses.size() == 0)
            break;

        //build a new graph with the remaining ones
        model_obj_matches_single_run = weightedGraphMatching(global_scene_hypotheses, avgFitness, ppf_params.avg_fitness_weight_thr_);
    }


    //set all unmatched current objects to NEW
    for (DetectedObject &co : object_vec_) {
        //curr object was not matched --> new or displaced on other plane
        if (co.state_ == ObjectState::UNKNOWN) {
            co.state_ = ObjectState::NEW;
        }
        curr_result.push_back(co);
    }

    //set all unmatched ref objects to REMOVED (this happens for the remaining part when a partial match was detected)
    for (DetectedObject &ro : model_vec_) {
        //ref object was not matched --> new or displaced on other plane
        if (ro.state_ == ObjectState::UNKNOWN) {
            ro.state_ = ObjectState::REMOVED;
        }
        ref_result.push_back(ro);
    }

    model_obj_matches.insert(model_obj_matches.end(), model_obj_matches_single_standing.begin(), model_obj_matches_single_standing.end());
    return model_obj_matches;
}

std::vector<Match> ObjectMatching::weightedGraphMatching(std::vector<ObjectHypothesesStruct> global_hypotheses,
                                                         std::function<float(FitnessScoreStruct)> computeFitness,
                                                         double fitness_thr) {

    boost::graph_traits< my_graph >::vertex_iterator vi, vi_end;

    my_graph g(0);

    std::map<std::string, vertex_t> modelToVertex;

    for (size_t o = 0; o < global_hypotheses.size(); o++) {
        std::string obj_vert_name= std::to_string(global_hypotheses[o].object_id)+"_object"; //make sure the vertex identifier is different from the model identifier (e.g. 2_mug)

        auto obj_vertex = boost::add_vertex(VertexProperty(obj_vert_name),g);

        const std::map<int, HypothesesStruct> &hypos = global_hypotheses[o].model_hyp;
        for (std::map<int, HypothesesStruct>::const_iterator it=hypos.begin(); it!=hypos.end(); ++it) {
            const HypothesesStruct h = it->second;
            std::string model_vert_name = std::to_string(h.model_id)+"_model";
            vertex_t model_vertex;
            if (modelToVertex.find(model_vert_name) == modelToVertex.end()) { //insert vertex into graph
                model_vertex = boost::add_vertex(VertexProperty(model_vert_name),g);
                modelToVertex.insert(std::make_pair(model_vert_name, model_vertex));
            } else {
                model_vertex = modelToVertex.find(model_vert_name)->second;
            }
            //boost::add_edge(o, model_uid, EdgeProperty(model_h->confidence_), g);
            //float ampl_weight = model_h->confidence_ * model_h->confidence_;

            float fitness = computeFitness(h.fitness);
            if(fitness > fitness_thr){
                float ampl_weight = std::exp(4 * fitness) - 1; //exp(0) = 1 -> -1;
                boost::add_edge(obj_vertex, model_vertex, EdgeProperty(ampl_weight, h), g); //edge with confidence and transformation between model and object
            }
        }
    }

    boost::print_graph(g);

    std::vector<Match> model_obj_match;

    std::vector< boost::graph_traits< my_graph >::vertex_descriptor > mate(boost::num_vertices(g));
    boost::brute_force_maximum_weighted_matching(g, &mate[0]);
    for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi) {
        if (mate[*vi] != boost::graph_traits< my_graph >::null_vertex()
                && boost::algorithm::contains(g[*vi].name, "model")) {//*vi < mate1[*vi]) {
            auto ed = boost::edge(*vi, mate[*vi], g); //returns pair<edge_descriptor, bool>, where bool indicates if edge exists or not
            float edge_weight = boost::get(boost::edge_weight_t(), g, ed.first);
            HypothesesStruct edge_hypo = boost::get(hypo_t(), g, ed.first);
            float deampl_weight = std::log(edge_weight + 1) / 4;
            std::cout << "{" << g[*vi].name << ", " << g[mate[*vi]].name << "} - " << deampl_weight << std::endl;

            //remove "_model" and "_object from the IDs (we added that before to be able to distinguish the nodes in the graph
            std::string m_name = g[*vi].name;
            std::string o_name = g[mate[*vi]].name;
            m_name.erase(m_name.length()-6, 6); //erase _model
            o_name.erase(o_name.length()-7, 7); //erase _object
            Match m(std::stoi(m_name), std::stoi(o_name), edge_hypo.transform, edge_hypo.fitness); //de-amplify
            model_obj_match.push_back(m);
        }
    }
    std::cout << std::endl;
    return model_obj_match;
}

//check if hypothesis is below floor (in hypotheses_verification.cpp a method exists using the convex hull)
//compute the z_value_threshold from the supporting plane
//checking for flying objects would not allow to keep true match of stacked objects
bool ObjectMatching::isBelowPlane(pcl::PointCloud<PointNormal>::ConstPtr model, pcl::PointCloud<PointNormal>::ConstPtr plane_cloud) {

    PointNormal minPoint, maxPoint;
    pcl::getMinMax3D(*model, minPoint, maxPoint);
    pcl::PointCloud<PointNormal>::Ptr cropped_object_plane(new pcl::PointCloud<PointNormal>);

    pcl::PassThrough<PointNormal> pass_plane_object;
    double object_plane_margin = 0.1;
    pass_plane_object.setInputCloud(plane_cloud);
    pass_plane_object.setFilterFieldName("x");
    pass_plane_object.setFilterLimits(minPoint.x - object_plane_margin, maxPoint.x + object_plane_margin);
    pass_plane_object.setKeepOrganized(false);
    pass_plane_object.filter(*cropped_object_plane);
    pass_plane_object.setInputCloud(cropped_object_plane);
    pass_plane_object.setFilterFieldName("y");
    pass_plane_object.setFilterLimits(minPoint.y - object_plane_margin, maxPoint.y + object_plane_margin);
    pass_plane_object.filter(*cropped_object_plane);

    int nr_plane_points=0;
    double plane_z_value_avg = 0.0;
    for (size_t i = 0; i < cropped_object_plane->size(); i++) {
        PointNormal &p = cropped_object_plane->points[i];
        if (pcl::isFinite(p)) { //it is a plane point
            plane_z_value_avg += p.z;
            nr_plane_points++;
        }
    }
    plane_z_value_avg /= nr_plane_points;

    pcl::PointCloud<PointNormal>::Ptr convex_hull_points(new pcl::PointCloud<PointNormal>);
    pcl::ConvexHull<PointNormal> convex_hull;
    convex_hull.setInputCloud(model);
    convex_hull.reconstruct(*convex_hull_points);
    const float min_z = plane_z_value_avg - 0.05;
    bool object_below_plane = std::any_of(convex_hull_points->begin(), convex_hull_points->end(),
                                          [min_z](const auto &pt) -> bool {
        return pt.z < min_z;
    });
    if (object_below_plane) {
        return true;
    }
    return false;
}



FitnessScoreStruct ObjectMatching::computeModelFitness(pcl::PointCloud<PointNormal>::ConstPtr object, pcl::PointCloud<PointNormal>::ConstPtr model,
                                                       v4r::apps::PPFRecognizerParameter param) {
    std::vector<v4r::ModelSceneCorrespondence> model_object_c;

    pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr object_octree;
    object_octree.reset(new pcl::octree::OctreePointCloudSearch<PointNormal>(0.02));
    object_octree->setInputCloud(object);
    object_octree->addPointsFromInputCloud();

    std::vector<float> nn_sqrd_distances;
    std::vector<int> nn_indices;

    std::vector<bool> object_overlapping_pts(object->size(), false);
    std::vector<bool> model_overlapping_pts(model->size(), false);

    for (size_t midx = 0; midx < model->size(); midx++) {
        PointNormal query_pt;
        query_pt.getVector4fMap() = model->at(midx).getVector4fMap();
        object_octree->radiusSearch(query_pt, param.hv_.inlier_threshold_xyz_, nn_indices, nn_sqrd_distances);

        const Eigen::Vector4f normal_m4f = model->points[midx].getNormalVector4fMap(); //sometimes the last value is nan
        const Eigen::Vector3f normal_m = normal_m4f.head<3>();

        for (size_t k = 0; k < nn_indices.size(); k++) {
            int sidx = nn_indices[k];

            model_overlapping_pts[midx] = true;
            object_overlapping_pts[sidx] = true;

            v4r::ModelSceneCorrespondence c(sidx, midx);
            const Eigen::Vector4f normal_s4f = object->points[sidx].getNormalVector4fMap();
            const Eigen::Vector3f normal_s = normal_s4f.head<3>();
            c.normals_dotp_ = normal_m.dot(normal_s);

            float normal_score = c.normals_dotp_ < param.hv_.inlier_threshold_normals_dotp_ ? 0.0 : c.normals_dotp_;
            //bool normal_score = c.normals_dotp_ > param.hv_.inlier_threshold_normals_dotp_;
            //bool color_score = true;
            float color_score = 1.0;
            if (!param.hv_.ignore_color_even_if_exists_) {
                Eigen::Vector3i  m_rgb =  model->points[midx].getRGBVector3i();
                Eigen::Vector3f color_m = rgb2lab(m_rgb);
                Eigen::Vector3i  o_rgb =  object->points[sidx].getRGBVector3i();
                Eigen::Vector3f color_o = rgb2lab(o_rgb);
                c.color_distance_ = v4r::computeCIEDE2000(color_o, color_m);
                //color_score = c.color_distance_ < param.hv_.inlier_threshold_color_;
                color_score = c.color_distance_ > param.hv_.inlier_threshold_color_ ? 0.0 :  1-(c.color_distance_ / param.hv_.inlier_threshold_color_);
            }
            if (normal_score == 0.0 || color_score == 0.0)
                c.fitness_ = 0.0;
            else
                c.fitness_ = 0.7*normal_score + 0.3*color_score;
            if (c.fitness_ != c.fitness_)
                std::cout << "fitness is nan" << std::endl;
            //c.fitness_ = normal_score * color_score; //either 0 or 1
            model_object_c.push_back(c);
        }
    }

    std::sort(model_object_c.begin(), model_object_c.end());

    Eigen::Array<bool, Eigen::Dynamic, 1> object_explained_pts(object->size());
    object_explained_pts.setConstant(object->size(), false);

    Eigen::Array<bool, Eigen::Dynamic, 1> model_explained_pts(model->size());
    model_explained_pts.setConstant(model->size(), false);

    Eigen::VectorXf modelFit = Eigen::VectorXf::Zero(model->size());
    Eigen::VectorXf objectFit = Eigen::VectorXf::Zero(object->size());

    std::vector<int> obj_expl_ind, model_expl_ind;

    for (const v4r::ModelSceneCorrespondence &c : model_object_c) {
        size_t oidx = c.scene_id_;
        size_t midx = c.model_id_;

        if (!object_explained_pts(oidx)) {
            object_explained_pts(oidx) = true;
            objectFit(oidx) = c.fitness_;
            if (c.fitness_ > 0.0)
                obj_expl_ind.push_back(oidx);
        }

        if (!model_explained_pts(midx)) {
            model_explained_pts(midx) = true;
            modelFit(midx) = c.fitness_;
            if (c.fitness_ > 0.0)
                model_expl_ind.push_back(midx);
        }
    }

    //don't divide by the number of all points, but only by the number of overlapping  points --> better results for combined objects?
    //--> no because the conficence for only partly overlapping objects is very big then

    int nr_model_overlapping_pts = 0, nr_object_overlappingt_pts = 0;
    for (size_t i = 0; i < model_overlapping_pts.size(); i++) {
        if (model_overlapping_pts[i]) {
            nr_model_overlapping_pts++;
        }
    }
    for (size_t i = 0; i < object_overlapping_pts.size(); i++) {
        if (object_overlapping_pts[i]) {
            nr_object_overlappingt_pts++;
        }
    }

    float model_fit = modelFit.sum();
    float model_confidence = model->empty() ? 0.f : model_fit /model->size(); //nr_model_overlapping_pts;

    float object_fit = objectFit.sum();
    float object_confidence = object->empty() ? 0.f : object_fit / object->size(); //nr_object_overlappingt_pts;

    FitnessScoreStruct fitness_struct(object_confidence, model_confidence, obj_expl_ind, model_expl_ind);

    return fitness_struct;
}

//estimate the distance between model and object
//transform the model to the object coordinate system, find point pairs and based on these compute the distance between the original model and the object
float ObjectMatching::estimateDistance(const pcl::PointCloud<PointNormal>::ConstPtr object_cloud, const pcl::PointCloud<PointNormal>::ConstPtr model_cloud, const Eigen::Matrix4f transform) {
    pcl::PointCloud<PointNormal>::Ptr model_transformed (new pcl::PointCloud<PointNormal>);
    pcl::transformPointCloudWithNormals(*model_cloud, *model_transformed, transform);

    pcl::KdTreeFLANN<PointNormal> kdtree;
    kdtree.setInputCloud (object_cloud);

    std::vector<int> pointIdxSearch;
    std::vector<float> pointSquaredDistance;

    float sum_eucl_dist = 0.0f;
    int nr_overlapping_pts = 0;
    for (size_t i = 0; i < model_transformed->size(); i++) {
        if ( kdtree.radiusSearch (model_transformed->points[i], 0.02, pointIdxSearch, pointSquaredDistance) > 0 ) {
            sum_eucl_dist += pcl::euclideanDistance(model_cloud->points[i], object_cloud->points[pointIdxSearch[0]]);
            nr_overlapping_pts++;
        }
    }
    return sum_eucl_dist/nr_overlapping_pts;
}

float ObjectMatching::computeMeanPointDistance(pcl::PointCloud<PointNormal>::ConstPtr ref_object, pcl::PointCloud<PointNormal>::ConstPtr curr_obj) {
    PointNormal ref_p_mean, curr_p_mean;
    ref_p_mean.x = ref_p_mean.y = ref_p_mean.z = 0.0f;
    curr_p_mean = ref_p_mean;

    for (size_t i = 0; i < ref_object->size(); i++) {
        ref_p_mean.x += ref_object->points[i].x;
        ref_p_mean.y += ref_object->points[i].y;
        ref_p_mean.z += ref_object->points[i].z;
    }
    ref_p_mean.x = ref_p_mean.x/ref_object->size();
    ref_p_mean.y = ref_p_mean.y/ref_object->size();
    ref_p_mean.z = ref_p_mean.z/ref_object->size();

    for (size_t i = 0; i < curr_obj->size(); i++) {
        curr_p_mean.x += curr_obj->points[i].x;
        curr_p_mean.y += curr_obj->points[i].y;
        curr_p_mean.z += curr_obj->points[i].z;
    }
    curr_p_mean.x = curr_p_mean.x/curr_obj->size();
    curr_p_mean.y = curr_p_mean.y/curr_obj->size();
    curr_p_mean.z = curr_p_mean.z/curr_obj->size();

    return pcl::euclideanDistance(ref_p_mean, curr_p_mean);
}

void ObjectMatching::saveCloudResults(pcl::PointCloud<PointNormal>::ConstPtr object_cloud, pcl::PointCloud<PointNormal>::ConstPtr model_aligned, std::string path) {

    //assign the matched model another label for better visualization
    pcl::PointXYZRGBL init_label_point;
    init_label_point.label=20;
    typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_object_aligned(new pcl::PointCloud<pcl::PointXYZRGBL>());
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_aligned_labeled(new pcl::PointCloud<pcl::PointXYZRGBL>(model_aligned->width, model_aligned->height, init_label_point));
    pcl::copyPointCloud(*model_aligned, *model_aligned_labeled);
    pcl::copyPointCloud(*object_cloud, *model_object_aligned);
    *model_object_aligned += *model_aligned_labeled;
    pcl::io::savePCDFileBinary(path +".pcd", *model_object_aligned);
}


//returns all valid clusters and indices that were removed are stored in filtered_ind
std::vector<pcl::PointIndices> ObjectMatching::clusterOutliersBySize(const pcl::PointCloud<PointNormal>::ConstPtr cloud, std::vector<int> &filtered_ind, float cluster_thr,
                                                                     int min_cluster_size, int max_cluster_size) {
    //clean up small things
    std::vector<pcl::PointIndices> cluster_indices;
    if (cloud->empty()) {
        return cluster_indices;
    }

    pcl::PointCloud<PointNormal>::Ptr cloud_copy(new pcl::PointCloud<PointNormal>);
    pcl::copyPointCloud(*cloud, *cloud_copy);
    cloud_copy->is_dense = false;

    //check if cloud only consists of nans
    std::vector<int> nan_ind;
    pcl::PointCloud<PointNormal>::Ptr no_nans_cloud(new pcl::PointCloud<PointNormal>);
    pcl::removeNaNFromPointCloud(*cloud_copy, *no_nans_cloud, nan_ind);
    if (no_nans_cloud->size() == 0) {
        return cluster_indices;
    } 

    pcl::EuclideanClusterExtraction<PointNormal> ec;
    ec.setClusterTolerance (cluster_thr);
    ec.setMinClusterSize (min_cluster_size);
    ec.setMaxClusterSize (max_cluster_size);
    ec.setInputCloud (no_nans_cloud);
    ec.extract (cluster_indices);

    //transform back to original indices
    for (pcl::PointIndices &ind : cluster_indices) {
        for (size_t i = 0; i < ind.indices.size(); i++) {
            ind.indices[i] = nan_ind[ind.indices[i]];
        }
    }

    //extract the indices that got filtered
    std::vector<int> cluster_ind;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        cluster_ind.insert(std::end(cluster_ind), std::begin(cluster_indices.at(i).indices), std::end(cluster_indices.at(i).indices));
    }
    //find points that are not nan and not in a cluster => filtered points
    for (size_t i = 0; i < cloud->size(); i++) {
        if (!pcl::isFinite(cloud->points[i]))
            continue;
        if (std::find(cluster_ind.begin(), cluster_ind.end(), i) == cluster_ind.end())
            filtered_ind.push_back(i);
    }
    return cluster_indices;
}



std::vector<v4r::ObjectHypothesesGroup> ObjectMatching::callRecognizer(DetectedObject &obj) {
    double ppf_rec_time_sum = 0.0;

    /// Create separate rgb and normal clouds in both cases to be able to call the recognizer
    pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointRGB>::Ptr object_rgb(new pcl::PointCloud<PointRGB>);

    pcl::copyPointCloud(*obj.getObjectCloud(), *object_rgb);
    pcl::copyPointCloud(*obj.getObjectCloud(), *object_normals);

    //choose the models that should be used with the recognizer
    std::vector<std::string> objects_to_look_for;
    for (DetectedObject m : model_vec_) {
        //if model object was already matched
        if (m.state_ == ObjectState::STATIC || m.state_ == ObjectState::DISPLACED)
            continue;

        //if we haven't tried to match this object with this model
        if (obj.already_checked_model_ids.find(m.getID()) == obj.already_checked_model_ids.end())
        {
            if (boost::filesystem::exists(m.object_folder_path_)) {
                objects_to_look_for.push_back(std::to_string(m.getID()));
                obj.already_checked_model_ids.insert(m.getID());
            }
        }
    }
    if (objects_to_look_for.size() == 0) { //otherwise the recognizer checks for all objects in the model folder
        std::cout << "No model that has not been checked." << obj.getID() << std::endl;
        return {};
    }

    //call the recognizer with object normals
    auto start = std::chrono::high_resolution_clock::now();
    auto hypothesis_groups = rec_->recognize(object_rgb, objects_to_look_for, object_normals);
    auto stop = std::chrono::high_resolution_clock::now();
    ppf_rec_time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

    std::cout << "Time spent to recognize objects with PPF: " << (ppf_rec_time_sum/1000) << " seconds." << std::endl;

    return hypothesis_groups;
}

std::pair<HypothesesStruct, bool> ObjectMatching::filterRecoHypothesis(DetectedObject obj, std::vector<v4r::ObjectHypothesis::Ptr> hg) {
    FitnessScoreStruct best_fitness;
    double best_score = 0.0;

    for (v4r::ObjectHypothesis::Ptr & h : hg) {
        //transform model
        typename pcl::PointCloud<PointNormal>::ConstPtr model_cloud = rec_->getModel(h->model_id_); //second paramater: resolution in mm
        typename pcl::PointCloud<PointNormal>::Ptr model_aligned(new pcl::PointCloud<PointNormal>());
        pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned, h->transform_);
        //tranformation after ICP was applied, if set in the config file
        typename pcl::PointCloud<PointNormal>::Ptr model_aligned_refined(new pcl::PointCloud<PointNormal>());
        pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned_refined, h->pose_refinement_ * h->transform_); /// pose_refinment contains ICP result

        //check if model and object are below supporting plane
        typename pcl::PointCloud<PointNormal>::Ptr object_aligned(new pcl::PointCloud<PointNormal>());
        pcl::transformPointCloudWithNormals(*obj.getObjectCloud(), *object_aligned, (h->pose_refinement_ * h->transform_).inverse()); /// pose_refinment contains ICP result
        int model_id = std::stoi(h->model_id_);
        std::vector<DetectedObject>::iterator ro_iter = std::find_if( model_vec_.begin(), model_vec_.end(),[&model_id](DetectedObject const &o) {return o.getID() == model_id; });
        if (isBelowPlane(model_aligned_refined, obj.plane_cloud_) && isBelowPlane(object_aligned, ro_iter->plane_cloud_)) {
            std::cout << "Model and object " << h->model_id_ << " are below plane and therefore not a valid solution" << std::endl;
            h->confidence_ = 0.0;
            continue;
        }

        //compute confidence based on normals and color between object and model
        FitnessScoreStruct fitness_score  = computeModelFitness(obj.getObjectCloud(), model_aligned_refined, ppf_params);

        //TODO: only object_conf or only model_conf?
        h->confidence_ = (fitness_score.object_conf + fitness_score.model_conf) / 2;
        std::cout << "Confidence " << obj.getID() << "-" << h->model_id_ << " " << h->confidence_ << std::endl;

        if (h->confidence_ > best_score)
        {
            best_score = h->confidence_;
            best_fitness = fitness_score;
        }
    }
    // do non-maxima surpression on all hypotheses in a hypotheses group based on model fitness (i.e. select only the
    // one hypothesis in group with best confidence score)
    std::sort(hg.begin(), hg.end(), [](const auto &a, const auto &b) { return a->confidence_ > b->confidence_; });
    hg.resize(1);

    if (hg[0]->confidence_ == 0.0) {
        //no valid solution found
        std::pair<HypothesesStruct, bool> return_val(HypothesesStruct(), false);
        return return_val;
    }

    typename pcl::PointCloud<PointNormal>::Ptr model_aligned_refined(new pcl::PointCloud<PointNormal>());
    pcl::transformPointCloudWithNormals(*(rec_->getModel(hg[0]->model_id_)), *model_aligned_refined, hg[0]->pose_refinement_ * hg[0]->transform_); /// pose_refinment contains ICP result

    HypothesesStruct hypo(std::stoi(hg[0]->model_id_), model_aligned_refined, hg[0]->pose_refinement_ * hg[0]->transform_, best_fitness);
    std::pair<HypothesesStruct, bool> return_val(hypo, true);
    return return_val;
}

std::vector<ObjectHypothesesStruct> ObjectMatching::createHypotheses() {
    std::vector<ObjectHypothesesStruct> global_scene_hypotheses;
    for (size_t i = 0; i < object_vec_.size(); i++) {

        if (object_vec_[i].state_ == ObjectState::STATIC || object_vec_[i].state_ == ObjectState::DISPLACED)
            continue;

        std::cout << "Perform PPF matching for object " << object_vec_[i].getID() << std::endl;

        std::vector<v4r::ObjectHypothesesGroup> hypothesis_groups = callRecognizer(object_vec_[i]);

        //remove hypothesis groups with 0 hypothesis
        hypothesis_groups.erase(std::remove_if(hypothesis_groups.begin(),hypothesis_groups.end(), [](const auto &a){ return a.ohs_.size() == 0; }), hypothesis_groups.end());

        if (hypothesis_groups.size() == 0) {
            std::cout << "New object. No hypothesis found for object " << object_vec_[i].getID() << std::endl;
            continue;
        }

        ObjectHypothesesStruct object_hypotheses;
        object_hypotheses.object_id = object_vec_[i].getID();
        object_hypotheses.object_cloud = object_vec_[i].getObjectCloud();


        boost::filesystem::path model_path_orig(model_path_);
        std::string cloud_matches_dir =  cloud_matches_dir_ + "/object" + std::to_string(object_vec_[i].getID());
        boost::filesystem::create_directories(cloud_matches_dir);
        pcl::io::savePCDFile(cloud_matches_dir + "/object.pcd", *object_vec_[i].getObjectCloud());

        // use results
        for (auto& hg : hypothesis_groups) {
            std::pair<HypothesesStruct, bool> hypo = filterRecoHypothesis(object_vec_[i], hg.ohs_);
            if (hypo.second) {
                object_hypotheses.model_hyp[hypo.first.model_id] = hypo.first;

                pcl::PointCloud<PointNormal>::Ptr model_aligned(new pcl::PointCloud<PointNormal>());
                pcl::transformPointCloudWithNormals(*(rec_->getModel(std::to_string(hypo.first.model_id))), *model_aligned, hypo.first.transform);

                std::string result_cloud_path = cloud_matches_dir + "/conf_" + std::to_string(hypo.first.fitness.object_conf) + "_" +
                        std::to_string(hypo.first.fitness.model_conf) + "_model_" + std::to_string(hypo.first.model_id) + "_" +
                        (ppf_params.ppf_rec_pipeline_.use_color_ ? "_color" : "");
                saveCloudResults(object_hypotheses.object_cloud, model_aligned, result_cloud_path);
            }
        }
        if (object_hypotheses.model_hyp.size() > 0 ) {
            global_scene_hypotheses.push_back(object_hypotheses);
        } else {
            std::cout << "New object. Couldn't find a hypothesis for object " << object_vec_[i].getID() << std::endl;
        }
    }
    return global_scene_hypotheses;
}

//starting from the good_pt_ids region growing is called
void ObjectMatching::matchedPartGrowing(pcl::PointCloud<PointNormal>::ConstPtr obj_cloud, pcl::PointCloud<PointNormal>::Ptr matched_part,
                                                                     pcl::PointCloud<PointNormal>::Ptr remaining_part, std::vector<int> good_pt_ids) {

    pcl::PointCloud<PointNormal>::Ptr good_pts(new pcl::PointCloud<PointNormal>);

    pcl::ExtractIndices<PointNormal> extract;
    extract.setInputCloud (obj_cloud);
    pcl::PointIndices::Ptr good_pt_ind(new pcl::PointIndices);
    good_pt_ind->indices = good_pt_ids;
    extract.setIndices (good_pt_ind);
    extract.setNegative (false);
    extract.setKeepOrganized(false);
    extract.filter(*good_pts);


    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud(*obj_cloud, *scene_normals);
    RegionGrowing<PointNormal, PointNormal> region_growing(obj_cloud, good_pts, scene_normals, false, 15.0, 5);
    std::vector<int> add_object_ind = region_growing.compute();

    pcl::PointCloud<PointNormal>::Ptr extracted_cloud(new pcl::PointCloud<PointNormal>);
    extract.setInputCloud (obj_cloud);
    pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
    obj_ind->indices = add_object_ind;
    extract.setIndices (obj_ind);
    extract.setNegative (false);
    extract.setKeepOrganized(false);
    extract.filter(*matched_part);

    extract.setNegative (true);
    extract.setKeepOrganized(false);
    extract.filter(*remaining_part);

    //remove very small clusters
    std::vector<int> small_cluster_ind;
    ObjectMatching::clusterOutliersBySize(remaining_part, small_cluster_ind, 0.014, min_object_size_ds);

    pcl::PointCloud<PointNormal>::Ptr small_cluster_cloud(new pcl::PointCloud<PointNormal>);
    extract.setInputCloud (remaining_part);
    obj_ind->indices = small_cluster_ind;
    extract.setIndices (obj_ind);
    extract.setNegative (false);
    extract.setKeepOrganized(false);
    extract.filter(*small_cluster_cloud);
    *matched_part += *small_cluster_cloud;

    extract.setNegative (true);
    extract.setKeepOrganized(false);
    extract.filter(*remaining_part);

    pcl::PointCloud<PointNormal>::Ptr ds_cloud = downsampleCloudVG(remaining_part, ds_leaf_size_ppf);
    if (!remaining_part->empty() && isObjectUnwanted(ds_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9)) {
        pcl::copyPointCloud(*obj_cloud, *matched_part);
        remaining_part->clear();
    }
}
