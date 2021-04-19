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
                               std::string model_path, std::string cfg_path) {
    model_vec_ = model_vec;
    object_vec_ = object_vec;
    model_path_ = model_path;
    cfg_path_ = cfg_path;
}

std::vector<Match> ObjectMatching::compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result)
{

    // setup recognizer options
    //---------------------------------------------------------------------------


    bf::path config_file = bf::path(cfg_path_);

    int verbosity = 0;
    bool force_retrain = false;  // if true, will retrain object models even if trained data already exists

    v4r::apps::PPFRecognizerParameter ppf_params;
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


    double ppf_rec_time_sum = 0.0;

    std::vector<ObjectHypothesesStruct> global_scene_hypotheses;


    auto start = std::chrono::high_resolution_clock::now();
    //omp_set_num_threads(1);
    v4r::apps::PPFRecognizer<pcl::PointXYZRGB> rec{ppf_params};
    rec.setModelsDir(model_path_);
    rec.setup(force_retrain);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(stop-start);
    std::cout << "Time to train the models " << duration.count() << " minutes" << std::endl;


    // use recognizer
    //---------------------------------------------------------------------------
    std::vector<std::string> objects_to_look_for{};   // empty vector means look for all objects

    for (size_t i = 0; i < object_vec_.size(); i++) {

        /// Create separate rgb and normal clouds in both cases to be able to call the recognizer
        pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<PointRGB>::Ptr object_rgb(new pcl::PointCloud<PointRGB>);

        pcl::copyPointCloud(*object_vec_[i].object_cloud_, *object_rgb);
        pcl::copyPointCloud(*object_vec_[i].object_cloud_, *object_normals);

        ObjectHypothesesStruct object_hypotheses;
        object_hypotheses.object_id = object_vec_[i].getID();
        object_hypotheses.object_cloud = object_vec_[i].object_cloud_;


        //call the recognizer with object normals
        auto start = std::chrono::high_resolution_clock::now();
        auto hypothesis_groups = rec.recognize(object_rgb, objects_to_look_for, object_normals);
        auto stop = std::chrono::high_resolution_clock::now();
        ppf_rec_time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
        //remove hypothesis groups with 0 hypothesis
        hypothesis_groups.erase(std::remove_if(hypothesis_groups.begin(),hypothesis_groups.end(), [](const auto &a){ return a.ohs_.size() == 0; }), hypothesis_groups.end());

        if (hypothesis_groups.size() == 0) {
            std::cout << "New object. No hypothesis found for object " << object_vec_[i].getID() << std::endl;
            continue;
        }

        //std::sort(hypothesis_groups.begin(), hypothesis_groups.end(), [](const auto& a, const auto& b) { return a.ohs_[0]->confidence_wo_hv_ > b.ohs_[0]->confidence_wo_hv_; });

        boost::filesystem::path model_path_orig(model_path_);
        std::string cloud_matches_dir =  model_path_orig.remove_trailing_separator().parent_path().string() + "/matches/object" + std::to_string(object_vec_[i].getID());
        boost::filesystem::create_directories(cloud_matches_dir);
        pcl::io::savePCDFile(cloud_matches_dir + "/object.pcd", *object_vec_[i].object_cloud_);
        // use results
        for (auto& hg : hypothesis_groups) {
            for (auto& h : hg.ohs_) {

                //transform model
                typename pcl::PointCloud<PointNormal>::ConstPtr model_cloud = rec.getModel(h->model_id_); //second paramater: resolution in mm
                typename pcl::PointCloud<PointNormal>::Ptr model_aligned(new pcl::PointCloud<PointNormal>());
                pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned, h->transform_);
                //tranformation after ICP was applied, if set in the config file
                typename pcl::PointCloud<PointNormal>::Ptr model_aligned_refined(new pcl::PointCloud<PointNormal>());
                pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned_refined, h->pose_refinement_ * h->transform_); /// pose_refinment contains ICP result

                //check if model is below supporting plane
                if (isModelBelowPlane(model_aligned_refined, object_vec_[i].plane_cloud_)) {
                    std::cout << "Model " << h->model_id_ << " is below plane and therefore not a valid solution" << std::endl;
                    continue;
                }

                //compute confidence based on normals and color between object and model
                object_hypotheses.obj_pts_not_explained_cloud = object_vec_[i].object_cloud_;
                std::tuple<float,float> obj_model_conf = computeModelFitness(object_vec_[i].object_cloud_, model_aligned_refined, ppf_params);

                //TODO: combined or max fitness score?
                //h->confidence_ = (std::get<0>(obj_model_conf) + std::get<1>(obj_model_conf)) / 2;
                h->confidence_ = std::max(std::get<0>(obj_model_conf), std::get<1>(obj_model_conf));
                std::cout << "Confidence " << object_hypotheses.object_id << "-" << h->model_id_ << " " << h->confidence_ << std::endl;

                std::string result_cloud_path = cloud_matches_dir + "/matchResult_model_" + h->model_id_ +"_conf_" + std::to_string(h->confidence_)+ "_" +
                        (ppf_params.ppf_rec_pipeline_.use_color_ ? "_color" : "");
                saveCloudResults(object_vec_[i].object_cloud_, model_aligned, model_aligned_refined, result_cloud_path);
            }
            // do non-maxima surpression on all hypotheses in a hypotheses group based on model fitness (i.e. select only the
            // one hypothesis in group with best confidence score)
            std::sort(hg.ohs_.begin(), hg.ohs_.end(), [](const auto &a, const auto &b) { return a->confidence_ > b->confidence_; });
            hg.ohs_.resize(1);

        }
        hypothesis_groups.erase(std::remove_if(hypothesis_groups.begin(),hypothesis_groups.end(), [&ppf_params](const auto &a){ return a.ohs_[0]->confidence_ < ppf_params.min_graph_conf_thr_ ; }), hypothesis_groups.end());

        if (hypothesis_groups.size() > 0 ) {
            object_hypotheses.hypotheses = hypothesis_groups;
            global_scene_hypotheses.push_back(object_hypotheses);
        } else {
            std::cout << "New object. Couldn't find a hypothesis for object " << object_vec_[i].getID() << std::endl;
        }
    }

    //use bipartite graph to do the matching
    //to be able to separate objects, do this as long as a result can be found
    std::map<int, pcl::PointCloud<PointNormal>::Ptr> model_id_cloud;
    for (size_t m = 0; m < model_vec_.size(); m++) {
        model_id_cloud.insert(std::make_pair(model_vec_[m].getID(), model_vec_[m].object_cloud_));
    }

    std::vector<Match> model_obj_matches;
    std::vector<Match> model_obj_matches_single_run = weightedGraphMatching(global_scene_hypotheses);
    while (model_obj_matches_single_run.size() > 0) {
        //remove already matched hypotheses
        model_obj_matches.insert(model_obj_matches.end(), model_obj_matches_single_run.begin(), model_obj_matches_single_run.end());
        for (size_t i = 0; i < model_obj_matches_single_run.size(); i++) {
            const int& obj_id_match = model_obj_matches_single_run[i].object_id;
            const int& model_id_match = model_obj_matches_single_run[i].model_id;
            auto object_hyp_iter = std::find_if( global_scene_hypotheses.begin(), global_scene_hypotheses.end(),[obj_id_match](ObjectHypothesesStruct const &o) {return o.object_id == obj_id_match; });
            if (object_hyp_iter == global_scene_hypotheses.end()) {
                std::cerr << "Wanted to remove a found match from the global hypotheses, but it is not there!? How can that happen?" << std::endl;
                return {}; //return empty vector
            }
            int scene_ind = std::distance(global_scene_hypotheses.begin(), object_hyp_iter);
            std::vector<v4r::ObjectHypothesesGroup>& hg = global_scene_hypotheses[scene_ind].hypotheses;
            auto model_hyp_iter = std::find_if(hg.begin(), hg.end(), [model_id_match](const auto &h) {return h.ohs_[0]->model_id_ == std::to_string(model_id_match);});

            //remove points from original object input cloud
            pcl::SegmentDifferences<PointNormal> diff;
            diff.setInputCloud(global_scene_hypotheses[scene_ind].obj_pts_not_explained_cloud);
            diff.setDistanceThreshold(0.02*0.02); //sqr_thr
            typename pcl::PointCloud<PointNormal>::ConstPtr model_cloud = rec.getModel((*model_hyp_iter).ohs_[0]->model_id_); //second paramater: resolution in mm
            typename pcl::PointCloud<PointNormal>::Ptr model_aligned_refined(new pcl::PointCloud<PointNormal>());
            pcl::copyPointCloud(*model_cloud, *model_aligned_refined);
            pcl::transformPointCloudWithNormals(*model_aligned_refined, *model_aligned_refined, (*model_hyp_iter).ohs_[0]->pose_refinement_ * (*model_hyp_iter).ohs_[0]->transform_);
            pcl::search::KdTree<PointNormal>::Ptr tree (new pcl::search::KdTree<PointNormal>);
            tree->setInputCloud(model_aligned_refined);
            diff.setSearchMethod(tree);
            diff.setTargetCloud(model_aligned_refined);
            pcl::PointCloud<PointNormal>::Ptr object_leftover (new pcl::PointCloud<PointNormal>());
            diff.segment(*object_leftover);

            //remove points from model_clouds
            pcl::transformPointCloudWithNormals(*(model_id_cloud.at(model_id_match)), *model_aligned_refined, (*model_hyp_iter).ohs_[0]->pose_refinement_ * (*model_hyp_iter).ohs_[0]->transform_);
            diff.setInputCloud(model_aligned_refined);
            tree->setInputCloud(global_scene_hypotheses[scene_ind].obj_pts_not_explained_cloud);
            diff.setSearchMethod(tree);
            diff.setTargetCloud(global_scene_hypotheses[scene_ind].obj_pts_not_explained_cloud);
            pcl::PointCloud<PointNormal>::Ptr model_leftover (new pcl::PointCloud<PointNormal>());
            diff.segment(*model_leftover);

            //transform the model back to original
            pcl::transformPointCloudWithNormals(*model_leftover, *model_leftover, ((*model_hyp_iter).ohs_[0]->pose_refinement_ * (*model_hyp_iter).ohs_[0]->transform_).inverse());

            model_id_cloud.at(model_id_match) = model_leftover;
            global_scene_hypotheses[scene_ind].obj_pts_not_explained_cloud = object_leftover;

            hg.erase(model_hyp_iter);
        }

        //First, erase all hypotheses where there are less than 100 points of the object not explained
        global_scene_hypotheses.erase(std::remove_if(global_scene_hypotheses.begin(), global_scene_hypotheses.end(), [](const ObjectHypothesesStruct oh) {return oh.obj_pts_not_explained_cloud->points.size() < 100; }), global_scene_hypotheses.end());
        if (global_scene_hypotheses.size() == 0) {
            break;
        }
        for (size_t gh = 0; gh < global_scene_hypotheses.size(); gh++) {
            ObjectHypothesesStruct &ohs = global_scene_hypotheses[gh];
            for (size_t h = 0; h < ohs.hypotheses.size(); h++) {
                //Second, update the confidences using the unexplained model points
                v4r::ObjectHypothesis::Ptr &oh = ohs.hypotheses[h].ohs_[0];
                int model_id = std::stoi(oh->model_id_);
                if (model_id_cloud.at(model_id)->size() < 100) {
                    oh->confidence_ = 0.0;
                }
                else {
                    typename pcl::PointCloud<PointNormal>::Ptr model_aligned_refined(new pcl::PointCloud<PointNormal>());
                    pcl::transformPointCloudWithNormals(*(model_id_cloud.at(model_id)), *model_aligned_refined, oh->pose_refinement_ * oh->transform_);
                    std::tuple<float,float> obj_model_conf = computeModelFitness(ohs.obj_pts_not_explained_cloud, model_aligned_refined, ppf_params);
                    //TODO: combined or max fitness score?
                    //h->confidence_ = (std::get<0>(obj_model_conf) + std::get<1>(obj_model_conf)) / 2;
                    oh->confidence_ = std::max(std::get<0>(obj_model_conf), std::get<1>(obj_model_conf));
                }
            }
            ohs.hypotheses.erase(std::remove_if(ohs.hypotheses.begin(),ohs.hypotheses.end(), [&ppf_params](const auto &a){ return a.ohs_[0]->confidence_ < ppf_params.min_graph_conf_thr_ ; }), ohs.hypotheses.end());
        }
        global_scene_hypotheses.erase(std::remove_if(global_scene_hypotheses.begin(), global_scene_hypotheses.end(), [](const ObjectHypothesesStruct oh) {return oh.hypotheses.size() == 0; }), global_scene_hypotheses.end());

        if (global_scene_hypotheses.size() == 0) {
            break;
        }

        //build a new graph with the remaining ones
        model_obj_matches_single_run = weightedGraphMatching(global_scene_hypotheses);
    }
    model_obj_matches.erase(std::remove_if(model_obj_matches.begin(),model_obj_matches.end(), [&ppf_params](const auto &a){ return a.confidence < ppf_params.min_result_conf_thr_ ; }), model_obj_matches.end());

    std::cout << "Time spent to recognize objects with PPF: " << (ppf_rec_time_sum/1000) << " seconds." << std::endl;

    //define the state of the detected objects based on the found matches
    std::map<int, DetectedObject*> ref_obj_map;
    std::map<int, DetectedObject*> curr_obj_map;
    for (DetectedObject &ro : model_vec_) {
        ref_obj_map.insert(std::make_pair(ro.getID(), &ro));
    }
    for (DetectedObject &co : object_vec_) {
        curr_obj_map.insert(std::make_pair(co.getID(), &co));
    }
    for (DetectedObject &ro : model_vec_) {
        auto m_iter = std::find_if( model_obj_matches.begin(), model_obj_matches.end(),[&ro](Match const &m) {return m.model_id == ro.getID(); });
        //a match was found
        //was it a partial match? was it displaced or static?
        if (m_iter != model_obj_matches.end()) {
            //get the original model and object clouds for the  match
            int obj_id = m_iter->object_id;
            DetectedObject* co = curr_obj_map.at(obj_id);
            const pcl::PointCloud<PointNormal>::Ptr model_cloud = ro.object_cloud_;
            const pcl::PointCloud<PointNormal>::Ptr object_cloud = co->object_cloud_;

            //compute distance between object and model
            float dist = computeDistance(object_cloud, model_cloud, m_iter->transform);
            bool is_static = dist < max_dist_for_being_static;

            pcl::PointCloud<PointNormal>::Ptr model_aligned(new pcl::PointCloud<PointNormal>());
            pcl::transformPointCloudWithNormals(*model_cloud, *model_aligned, m_iter->transform);

            //remove points from MODEL object input cloud
            SceneDifferencingPoints scene_diff(0.02);
            std::vector<int> diff_ind;
            std::vector<int> corresponding_ind;
            SceneDifferencingPoints::Cloud::Ptr model_diff_cloud = scene_diff.computeDifference(model_aligned, object_cloud, diff_ind, corresponding_ind);

            if (diff_ind.size() < 100) { //if less than 100 points left, we do not split the model object
                if (is_static) {
                    ro.state_ = ObjectState::STATIC;
                    ro.match_ = *m_iter;
                } else {
                    ro.state_ = ObjectState::DISPLACED;
                    ro.match_ = *m_iter;
                }
                ref_result.push_back(ro);
            }
            //split the model cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_model_part(new pcl::PointCloud<PointNormal>);
                pcl::ExtractIndices<PointNormal> extract;
                extract.setInputCloud (model_aligned);
                pcl::PointIndices::Ptr c_ind(new pcl::PointIndices());
                c_ind->indices = corresponding_ind;
                extract.setIndices(c_ind);
                extract.setKeepOrganized(false);
                extract.setNegative (false);
                extract.filter(*matched_model_part);
                DetectedObject model_part(matched_model_part, ro.plane_cloud_, ro.supp_plane_, (is_static ? ObjectState::STATIC : ObjectState::DISPLACED), "");
                ref_result.push_back((model_part));

                //the remaining part of the model
                ro.object_cloud_=model_diff_cloud;
                ro.state_ = ObjectState::REMOVED;
                ro.object_folder_path_="";
            }

            //remove points from OBJECT object input cloud
            pcl::PointCloud<PointNormal>::Ptr object_diff_cloud = scene_diff.computeDifference(object_cloud, model_aligned, diff_ind, corresponding_ind);

            if (diff_ind.size() < 100) { //if less than 100 points left, we do not split the object
                if (is_static) {
                    co->state_ = ObjectState::STATIC;
                    co->match_ = *m_iter;
                } else {
                    co->state_ = ObjectState::DISPLACED;
                    co->match_ = *m_iter;
                }
                curr_result.push_back(*co);
            }
            //split the object cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_object_part(new pcl::PointCloud<PointNormal>);
                pcl::ExtractIndices<PointNormal> extract;
                pcl::PointIndices::Ptr c_ind(new pcl::PointIndices());
                extract.setInputCloud (object_cloud);
                c_ind->indices = corresponding_ind;
                extract.setIndices(c_ind);
                extract.setKeepOrganized(false);
                extract.setNegative (false);
                extract.filter(*matched_object_part);
                DetectedObject object_part(matched_object_part, co->plane_cloud_, co->supp_plane_, (is_static ? ObjectState::STATIC : ObjectState::DISPLACED), "");
                curr_result.push_back(object_part);

                //the remaining part of the object
                co->object_cloud_=object_diff_cloud;
                co->state_ = ObjectState::NEW;
                co->object_folder_path_="";
            }
        }
    }

    //set all unmatched current objects to NEW
    for (DetectedObject &co : object_vec_) {
        auto m_iter = std::find_if( model_obj_matches.begin(), model_obj_matches.end(),[&co](Match const &m) {return m.object_id == co.getID(); });
        //curr object was not matched --> new or displaced on other plane
        if (m_iter == model_obj_matches.end() || co.state_ == ObjectState::UNKNOWN) {
            co.state_ = ObjectState::NEW;
            curr_result.push_back(co);
        }
    }

    //set all unmatched ref objects to REMOVED (this happens for the remaining part when a partial match was detected)
    for (DetectedObject &ro : model_vec_) {
        auto m_iter = std::find_if( model_obj_matches.begin(), model_obj_matches.end(),[&ro](Match const &m) {return m.model_id == ro.getID(); });
        //curr object was not matched --> new or displaced on other plane
        if (m_iter == model_obj_matches.end() || ro.state_ == ObjectState::UNKNOWN) {
            ro.state_ = ObjectState::REMOVED;
            ref_result.push_back(ro);
        }
    }

    return model_obj_matches;
}

std::vector<Match> ObjectMatching::weightedGraphMatching(std::vector<ObjectHypothesesStruct> global_hypotheses) {

    boost::graph_traits< my_graph >::vertex_iterator vi, vi_end;

    my_graph g(0);

    std::map<std::string, vertex_t> modelToVertex;

    for (size_t o = 0; o < global_hypotheses.size(); o++) {
        std::string obj_vert_name= std::to_string(global_hypotheses[o].object_id)+"_object"; //make sure the vertex identifier is different from the model identifier (e.g. 2_mug)

        auto obj_vertex = boost::add_vertex(VertexProperty(obj_vert_name),g);

        for (size_t h = 0; h < global_hypotheses[o].hypotheses.size(); h++) {
            const v4r::ObjectHypothesesGroup &obj_hypothesis = global_hypotheses[o].hypotheses[h];
            const v4r::ObjectHypothesis::Ptr &model_h = obj_hypothesis.ohs_[0];
            std::string model_vert_name = model_h->model_id_+"_model";
            vertex_t model_vertex;
            if (modelToVertex.find(model_vert_name) == modelToVertex.end()) { //insert vertex into graph
                model_vertex = boost::add_vertex(VertexProperty(model_vert_name),g);
                modelToVertex.insert(std::make_pair(model_vert_name, model_vertex));
            } else {
                model_vertex = modelToVertex.find(model_vert_name)->second;
            }
            //boost::add_edge(o, model_uid, EdgeProperty(model_h->confidence_), g);
            float ampl_weight = model_h->confidence_ * model_h->confidence_;
            Eigen::Matrix4f transformation = model_h->pose_refinement_ * model_h->transform_;
            boost::add_edge(obj_vertex, model_vertex, EdgeProperty(ampl_weight, transformation), g); //edge with confidence and transformation between model and object
        }
    }

    //boost::print_graph(g);

    std::vector<Match> model_obj_match;

    std::vector< boost::graph_traits< my_graph >::vertex_descriptor > mate(boost::num_vertices(g));
    boost::brute_force_maximum_weighted_matching(g, &mate[0]);
    for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi) {
        if (mate[*vi] != boost::graph_traits< my_graph >::null_vertex()
                && boost::algorithm::contains(g[*vi].name, "model")) {//*vi < mate1[*vi]) {
            auto ed = boost::edge(*vi, mate[*vi], g); //returns pair<edge_descriptor, bool>, where bool indicates if edge exists or not
            float edge_weight = boost::get(boost::edge_weight_t(), g, ed.first);
            Eigen::Matrix4f edge_transformation = boost::get(transformation_t(), g, ed.first);
            std::cout << "{" << g[*vi].name << ", " << g[mate[*vi]].name << "} - " << edge_weight << std::endl;

            //remove "_model" and "_object from the IDs (we added that before to be able to distinguish the nodes in the graph
            std::string m_name = g[*vi].name;
            std::string o_name = g[mate[*vi]].name;
            m_name.erase(m_name.length()-6, 6); //erase _model
            o_name.erase(o_name.length()-7, 7); //erase _object
            Match m(std::stoi(m_name), std::stoi(o_name), std::sqrt(edge_weight), edge_transformation); //de-amplify
            model_obj_match.push_back(m);
        }
    }
    std::cout << std::endl;
    return model_obj_match;
}

//check if hypothesis is below floor (in hypotheses_verification.cpp a method exists using the convex hull)
//compute the z_value_threshold from the supporting plane
//checking for flying objects would not allow to keep true match of stacked objects
bool ObjectMatching::isModelBelowPlane(pcl::PointCloud<PointNormal>::Ptr model, pcl::PointCloud<PointNormal>::Ptr plane_cloud) {

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

    int nr_plane_points;
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



std::tuple<float,float> ObjectMatching::computeModelFitness(pcl::PointCloud<PointNormal>::Ptr object, pcl::PointCloud<PointNormal>::Ptr model,
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

        const auto normal_m = model->points[midx].getNormalVector4fMap();

        for (size_t k = 0; k < nn_indices.size(); k++) {
            int sidx = nn_indices[k];

            model_overlapping_pts[midx] = true;
            object_overlapping_pts[sidx] = true;

            v4r::ModelSceneCorrespondence c(sidx, midx);
            const auto normal_s = object->at(sidx).getNormalVector4fMap();
            c.normals_dotp_ = normal_m.dot(normal_s);

            bool normal_score = c.normals_dotp_ > param.hv_.inlier_threshold_normals_dotp_;
            bool color_score = true;
            if (!param.hv_.ignore_color_even_if_exists_) {
                Eigen::Vector3i  m_rgb =  model->points[midx].getRGBVector3i();
                Eigen::Vector3f color_m = rgb2lab(m_rgb);
                Eigen::Vector3i  o_rgb =  object->points[sidx].getRGBVector3i();
                Eigen::Vector3f color_o = rgb2lab(o_rgb);
                c.color_distance_ = v4r::computeCIEDE2000(color_o, color_m);
                color_score = c.color_distance_ < param.hv_.inlier_threshold_color_;
            }
            c.fitness_ = normal_score * color_score; //either 0 or 1
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

    for (const v4r::ModelSceneCorrespondence &c : model_object_c) {
        size_t oidx = c.scene_id_;
        size_t midx = c.model_id_;

        if (!object_explained_pts(oidx)) {
            object_explained_pts(oidx) = true;
            objectFit(oidx) = c.fitness_;
        }

        if (!model_explained_pts(midx)) {
            model_explained_pts(midx) = true;
            modelFit(midx) = c.fitness_;
        }
    }

    //don't divide by the number of all points, but only by the number of overlapping  points --> better results for combined objects?
    //--> no because the conficence for only partly overlapping objects is very big then
    int nr_model_overlapping_pts = 0, nr_object_overlappingt_pts = 0;
    for (size_t i = 0; i < model_overlapping_pts.size(); i++) {
        if (model_overlapping_pts[i])
            nr_model_overlapping_pts++;
    }
    for (size_t i = 0; i < object_overlapping_pts.size(); i++) {
        if (object_overlapping_pts[i])
            nr_object_overlappingt_pts++;
    }

    float model_fit = modelFit.sum();
    float model_confidence = model->empty() ? 0.f : model_fit /model->size(); //nr_model_overlapping_pts;

    float object_fit = objectFit.sum();
    float object_confidence = object->empty() ? 0.f : object_fit / object->size(); //nr_object_overlappingt_pts;

    return std::make_tuple(object_confidence,model_confidence);
}

//estimate the distance between model and object
//transform the model to the object coordinate system, find point pairs and based on these compute the distance between the original model and the object
float ObjectMatching::computeDistance(const pcl::PointCloud<PointNormal>::Ptr object_cloud, const pcl::PointCloud<PointNormal>::Ptr model_cloud, const Eigen::Matrix4f transform) {
    pcl::PointCloud<PointNormal>::Ptr model_transformed (new pcl::PointCloud<PointNormal>);
    pcl::transformPointCloudWithNormals(*model_cloud, *model_transformed, transform);

    pcl::KdTreeFLANN<PointNormal> kdtree;
    kdtree.setInputCloud (object_cloud);

    std::vector<int> pointIdxSearch;
    std::vector<float> pointSquaredDistance;

    float sum_eucl_dist = 0.0f;
    for (size_t i = 0; i < model_transformed->size(); i++) {
        if ( kdtree.radiusSearch (model_transformed->points[i], 0.02, pointIdxSearch, pointSquaredDistance) > 0 ) {
            sum_eucl_dist += pcl::euclideanDistance(model_cloud->points[i], object_cloud->points[pointIdxSearch[0]]);
        }
    }
    return sum_eucl_dist/model_transformed->size();
}

void ObjectMatching::saveCloudResults(pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr model_aligned,
                                      pcl::PointCloud<PointNormal>::Ptr model_aligned_refined, std::string path) {

    //assign the matched model another label for better visualization
    pcl::PointXYZRGBL init_label_point;
    init_label_point.label=10;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_algined_ref_labeled(new pcl::PointCloud<pcl::PointXYZRGBL>(model_aligned_refined->width, model_aligned_refined->height, init_label_point));
    pcl::copyPointCloud(*model_aligned_refined, * model_algined_ref_labeled);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_object_after_icp{new pcl::PointCloud<pcl::PointXYZRGBL>};
    pcl::copyPointCloud(*object_cloud, *model_object_after_icp);
    *model_object_after_icp += *model_algined_ref_labeled;
    pcl::io::savePCDFileBinary(path +".pcd", *model_object_after_icp);

    //show the result of ICP
    init_label_point.label=20;
    typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_object_before_icp(new pcl::PointCloud<pcl::PointXYZRGBL>());
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr model_aligned_labeled(new pcl::PointCloud<pcl::PointXYZRGBL>(model_aligned->width, model_aligned->height, init_label_point));
    pcl::copyPointCloud(*model_aligned, *model_aligned_labeled);
    pcl::copyPointCloud(*object_cloud, *model_object_before_icp);
    *model_object_before_icp += *model_aligned_labeled;
    pcl::io::savePCDFileBinary(path +"_before_icp.pcd", *model_object_before_icp);
}


