
#include <PPFRecognizer.h>
#include <v4r/recognition/object_hypothesis.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZRGB PointT;

int main(int argc, char** argv)
{
    std::vector<std::string> arguments{argv + 1, argv + argc};

    // setup recognizer options
    //---------------------------------------------------------------------------

    std::cout << "Initializing recognizer with: " << std::endl;
    for (auto& arg : arguments)
        std::cout << arg << " ";
    std::cout << std::endl;

    // get path relative to executable location (independent of working directory)
    //  std::string cfg_dir;
    //  char path[1024];
    //  ssize_t nRead = readlink("/proc/self/exe", path, sizeof(path)-1);
    //  if (nRead != -1) {
    //    path[nRead] = '\0';
    //    cfg_dir = std::string{path} + "../cfg";
    //  } else {
    //    std::cerr << "Could not compute cfg directory location" << std::endl;
    //    return -1;
    //  }


    bf::path models_dir;
    bf::path config_file = bf::path("../../ppf/cfg/")/"ppf_pose_estimation_config.ini";

    int verbosity = 0;
    bool visualize = false;
    bool ignore_ROI_from_file = false;
    bool ask_for_ROI = false;
    bool force_retrain = true;  // if true, will retrain object models even if trained data already exists
    bool use_color = true;
    v4r::apps::PPFRecognizerParameter ppf_params;

    po::options_description desc("PPF Object Instance Recognizer\n"
                                 "==============================\n"
                                 "     **Allowed options**\n");

    // get config file or use default
    desc.add_options()
            ("help,h", "produce help message")
            ("cfg,c", po::value<bf::path>(&config_file)->default_value(config_file),
             "File path of V4R config (.ini) file containing parameters for the recognition pipeline");

    po::variables_map vm;
    po::parsed_options parsed_tmp = po::command_line_parser(arguments)
            .options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further =
            po::collect_unrecognized(parsed_tmp.options, po::include_positional);
    po::store(parsed_tmp, vm);

    try {
        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
    }

    desc.add_options()
            ("model_dir,m", po::value<bf::path>(&models_dir)->required(), "Directory with object models.")
            ("verbosity", po::value<int>(&verbosity)->default_value(verbosity),
             "set verbosity level for output (<0 minimal output)")
            ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
            ("ignore_ROI_from_file", po::bool_switch(&ignore_ROI_from_file),
             "if set, does not try to read ROI from file")
            ("ask_for_ROI", po::bool_switch(&ask_for_ROI), "if true, asks the user to provide ROI")
            ("retrain", po::bool_switch(&force_retrain),
             "If set, retrains the object models no matter if they already exists.");

    ppf_params.init(desc);
    po::parsed_options parsed = po::command_line_parser(to_pass_further).options(desc).run();
    po::store(parsed, vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
    }

    if (v4r::io::existsFile(config_file)) {
        std::ifstream f(config_file.string());
        po::parsed_options config_parsed = po::parse_config_file(f, desc);
        po::store(config_parsed, vm);
        f.close();
    } else {
        std::cerr << config_file.string() << " does not exist! Usage: " << desc;
    }

    try {
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    if (verbosity >= 0) {
        FLAGS_v = verbosity;
        std::cout << "Enabling verbose logging." << std::endl;
    } else {
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }
    google::InitGoogleLogging(argv[0]);


    v4r::apps::PPFRecognizer<pcl::PointXYZRGB> rec{ppf_params};
    rec.setModelsDir(models_dir);
    rec.setup(force_retrain, use_color);

    // use recognizer
    //---------------------------------------------------------------------------
    std::vector<std::string> objects_to_look_for{};   // empty vector means look for all objects

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene{new pcl::PointCloud<pcl::PointXYZRGB>};
    std::string scene_filename{"../../ppf/models/manipulation-scene.pcd"};
    if (pcl::io::loadPCDFile(scene_filename, *scene) < 0) {
        std::cerr << "Couldn't read the file: " << scene_filename << "\n";
        return -1;
    }

    auto hypothesis_groups = rec.recognize(scene, objects_to_look_for);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_with_objects{new pcl::PointCloud<pcl::PointXYZRGBNormal>};
    pcl::copyPointCloud(*scene, *scene_with_objects);

    // use results
    for (const auto& hg : hypothesis_groups) {
        for (const auto& h : hg.ohs_) {
            if (h->is_verified_) {
                std::cout << "model: "      << h->model_id_         << "\n"
                          << "confidence: " << h->confidence_wo_hv_ << "\n"
                          << "transform:\n"  << h->transform_        << "\n\n";

                //transform object model
                typename pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr model_cloud = rec.getModel(h->model_id_, 3);
                typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_aligned(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_aligned_refined(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                pcl::transformPointCloud(*model_cloud, *model_aligned, h->transform_);
                pcl::transformPointCloud(*model_cloud, *model_aligned_refined, h->pose_refinement_ * h->transform_);

                // assign unique color for each object hypothesis
                const uint8_t r = rand() % 255;
                const uint8_t g = rand() % 255;
                const uint8_t b = rand() % 255;

                for (size_t i = 0; i < model_aligned_refined->size(); i++) {
                    model_aligned_refined->points[i].r=r;
                    model_aligned_refined->points[i].g=g;
                    model_aligned_refined->points[i].b=b;
                }
                *scene_with_objects += *model_aligned_refined;
            }
        }
    }

    pcl::io::savePCDFileBinary("../../ppf/models/scene_with_objects.pcd", *scene_with_objects);

    return 0;
}
