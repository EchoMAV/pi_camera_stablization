#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videostab.hpp>
//#include <opencv2/highgui.hpp>

namespace
{

    cv::Ptr<cv::videostab::IFrameSource> stabilizedFrames;
    std::string saveMotionsPath;
    double outputFps;
    std::string outputPath;

    cv::videostab::MotionModel motionModel(std::string const &str)
    {
        if (str == "transl")
        {
            return cv::videostab::MM_TRANSLATION;
        }
        if (str == "transl_and_scale")
        {
            return cv::videostab::MM_TRANSLATION_AND_SCALE;
        }
        if (str == "rigid")
        {
            return cv::videostab::MM_RIGID;
        }
        if (str == "similarity")
        {
            return cv::videostab::MM_SIMILARITY;
        }
        if (str == "affine")
        {
            return cv::videostab::MM_AFFINE;
        }
        if (str == "homography")
        {
            return cv::videostab::MM_HOMOGRAPHY;
        }
        throw std::runtime_error("unknown motion model: " + str);
    }

    class IMotionEstimatorBuilder
    {
    public:
        virtual ~IMotionEstimatorBuilder() {}
        virtual cv::Ptr<cv::videostab::ImageMotionEstimatorBase> build() = 0;

    protected:
        IMotionEstimatorBuilder(cv::CommandLineParser &command) : m_cmd(command) {}
        cv::CommandLineParser m_cmd;
    };

    class MotionEstimatorRansacL2Builder : public IMotionEstimatorBuilder
    {
    public:
        MotionEstimatorRansacL2Builder(cv::CommandLineParser &command, bool use_gpu, const std::string &_prefix = "")
            : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

        virtual cv::Ptr<cv::videostab::ImageMotionEstimatorBase> build()
        {
            cv::Ptr<cv::videostab::MotionEstimatorRansacL2> est = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(motionModel(m_cmd.get<std::string>(prefix + "model")));

            cv::videostab::RansacParams ransac = est->ransacParams();
            if (m_cmd.get<std::string>(prefix + "subset") != "auto")
                ransac.size = m_cmd.get<int>(prefix + "subset");
            if (m_cmd.get<std::string>(prefix + "thresh") != "auto")
                ransac.thresh = m_cmd.get<float>(prefix + "thresh");
            ransac.eps = m_cmd.get<float>(prefix + "outlier-ratio");
            est->setRansacParams(ransac);

            est->setMinInlierRatio(m_cmd.get<float>(prefix + "min-inlier-ratio"));

            cv::Ptr<cv::videostab::IOutlierRejector> outlierRejector = cv::makePtr<cv::videostab::NullOutlierRejector>();
            if (m_cmd.get<std::string>(prefix + "local-outlier-rejection") == "yes")
            {
                cv::Ptr<cv::videostab::TranslationBasedLocalOutlierRejector> tblor = cv::makePtr<cv::videostab::TranslationBasedLocalOutlierRejector>();
                cv::videostab::RansacParams ransacParams = tblor->ransacParams();
                if (m_cmd.get<std::string>(prefix + "thresh") != "auto")
                    ransacParams.thresh = m_cmd.get<float>(prefix + "thresh");
                tblor->setRansacParams(ransacParams);
                outlierRejector = tblor;
            }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
            if (gpu)
            {
                Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
                kbest->setOutlierRejector(outlierRejector);
                return kbest;
            }
#endif

            cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);
            kbest->setDetector(cv::GFTTDetector::create(m_cmd.get<int>(prefix + "nkps")));
            kbest->setOutlierRejector(outlierRejector);
            return kbest;
        }

    private:
        bool gpu;
        std::string prefix;
    };

    class MotionEstimatorL1Builder : public IMotionEstimatorBuilder
    {
    public:
        MotionEstimatorL1Builder(cv::CommandLineParser &command, bool use_gpu, const std::string &_prefix = "")
            : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

        virtual cv::Ptr<cv::videostab::ImageMotionEstimatorBase> build()
        {
            cv::Ptr<cv::videostab::MotionEstimatorL1> est = cv::makePtr<cv::videostab::MotionEstimatorL1>(motionModel(m_cmd.get<std::string>(prefix + "model")));

            cv::Ptr<cv::videostab::IOutlierRejector> outlierRejector = cv::makePtr<cv::videostab::NullOutlierRejector>();
            if (m_cmd.get<std::string>(prefix + "local-outlier-rejection") == "yes")
            {
                cv::Ptr<cv::videostab::TranslationBasedLocalOutlierRejector> tblor = cv::makePtr<cv::videostab::TranslationBasedLocalOutlierRejector>();
                cv::videostab::RansacParams ransacParams = tblor->ransacParams();
                if (m_cmd.get<std::string>(prefix + "thresh") != "auto")
                    ransacParams.thresh = m_cmd.get<float>(prefix + "thresh");
                tblor->setRansacParams(ransacParams);
                outlierRejector = tblor;
            }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
            if (gpu)
            {
                Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
                kbest->setOutlierRejector(outlierRejector);
                return kbest;
            }
#endif

            cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> kbest = cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(est);
            kbest->setDetector(cv::GFTTDetector::create(m_cmd.get<int>(prefix + "nkps")));
            kbest->setOutlierRejector(outlierRejector);
            return kbest;
        }

    private:
        bool gpu;
        std::string prefix;
    };

    void run()
    {
        cv::VideoWriter writer;
        cv::Mat stabilizedFrame;
        int nframes = 0;
        char file_name[100];

        // for each stabilized frame
        while (!(stabilizedFrame = stabilizedFrames->nextFrame()).empty())
        {
            nframes++;

            // init writer (once) and save stabilized frame
            if (!outputPath.empty())
            {
                if (!writer.isOpened())
                    writer.open(outputPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                                outputFps, stabilizedFrame.size());
                writer << stabilizedFrame;
            }

            // show stabilized frame
        }

        std::cout << "processed frames: " << nframes << std::endl
                  << "finished\n";
    }

    void printHelp()
    {
        std::cout << "OpenCV video stabilizer.\n"
                     "Usage: videostab <file_path> [arguments]\n\n"
                     "Arguments:\n"
                     "  -m=, --model=(transl|transl_and_scale|rigid|similarity|affine|homography)\n"
                     "      Set motion model. The default is affine.\n"
                     "  -lp=, --lin-prog-motion-est=(yes|no)\n"
                     "      Turn on/off LP based motion estimation. The default is no.\n"
                     "  --subset=(<int_number>|auto)\n"
                     "      Number of random samples per one motion hypothesis. The default is auto.\n"
                     "  --thresh=(<float_number>|auto)\n"
                     "      Maximum error to classify match as inlier. The default is auto.\n"
                     "  --outlier-ratio=<float_number>\n"
                     "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
                     "  --min-inlier-ratio=<float_number>\n"
                     "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
                     "  --nkps=<int_number>\n"
                     "      Number of keypoints to find in each frame. The default is 1000.\n"
                     "  --local-outlier-rejection=(yes|no)\n"
                     "      Perform local outlier rejection. The default is no.\n\n"
                     "  -sm=, --save-motions=(<file_path>|no)\n"
                     "      Save estimated motions into file. The default is no.\n"
                     "  -lm=, --load-motions=(<file_path>|no)\n"
                     "      Load motions from file. The default is no.\n\n"
                     "  -r=, --radius=<int_number>\n"
                     "      Set sliding window radius. The default is 15.\n"
                     "  --stdev=(<float_number>|auto)\n"
                     "      Set smoothing weights standard deviation. The default is auto\n"
                     "      (i.e. sqrt(radius)).\n"
                     "  -lps=, --lin-prog-stab=(yes|no)\n"
                     "      Turn on/off linear programming based stabilization method.\n"
                     "  --lps-trim-ratio=(<float_number>|auto)\n"
                     "      Trimming ratio used in linear programming based method.\n"
                     "  --lps-w1=(<float_number>|1)\n"
                     "      1st derivative weight. The default is 1.\n"
                     "  --lps-w2=(<float_number>|10)\n"
                     "      2nd derivative weight. The default is 10.\n"
                     "  --lps-w3=(<float_number>|100)\n"
                     "      3rd derivative weight. The default is 100.\n"
                     "  --lps-w4=(<float_number>|100)\n"
                     "      Non-translation motion components weight. The default is 100.\n\n"
                     "  --deblur=(yes|no)\n"
                     "      Do deblurring.\n"
                     "  --deblur-sens=<float_number>\n"
                     "      Set deblurring sensitivity (from 0 to +inf). The default is 0.1.\n\n"
                     "  -t=, --trim-ratio=<float_number>\n"
                     "      Set trimming ratio (from 0 to 0.5). The default is 0.1.\n"
                     "  -et=, --est-trim=(yes|no)\n"
                     "      Estimate trim ratio automatically. The default is yes.\n"
                     "  -ic=, --incl-constr=(yes|no)\n"
                     "      Ensure the inclusion constraint is always satisfied. The default is no.\n\n"
                     "  -bm=, --border-mode=(replicate|reflect|const)\n"
                     "      Set border extrapolation mode. The default is replicate.\n\n"
                     "  --mosaic=(yes|no)\n"
                     "      Do consistent mosaicing. The default is no.\n"
                     "  --mosaic-stdev=<float_number>\n"
                     "      Consistent mosaicing stdev threshold. The default is 10.0.\n\n"
                     "  -mi=, --motion-inpaint=(yes|no)\n"
                     "      Do motion inpainting (requires CUDA support). The default is no.\n"
                     "  --mi-dist-thresh=<float_number>\n"
                     "      Estimated flow distance threshold for motion inpainting. The default is 5.0.\n\n"
                     "  -ci=, --color-inpaint=(no|average|ns|telea)\n"
                     "      Do color inpainting. The defailt is no.\n"
                     "  --ci-radius=<float_number>\n"
                     "      Set color inpainting radius (for ns and telea options only).\n"
                     "      The default is 2.0\n\n"
                     "  -ws=, --wobble-suppress=(yes|no)\n"
                     "      Perform wobble suppression. The default is no.\n"
                     "  --ws-lp=(yes|no)\n"
                     "      Turn on/off LP based motion estimation. The default is no.\n"
                     "  --ws-period=<int_number>\n"
                     "      Set wobble suppression period. The default is 30.\n"
                     "  --ws-model=(transl|transl_and_scale|rigid|similarity|affine|homography)\n"
                     "      Set wobble suppression motion model (must have more DOF than motion \n"
                     "      estimation model). The default is homography.\n"
                     "  --ws-subset=(<int_number>|auto)\n"
                     "      Number of random samples per one motion hypothesis. The default is auto.\n"
                     "  --ws-thresh=(<float_number>|auto)\n"
                     "      Maximum error to classify match as inlier. The default is auto.\n"
                     "  --ws-outlier-ratio=<float_number>\n"
                     "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
                     "  --ws-min-inlier-ratio=<float_number>\n"
                     "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
                     "  --ws-nkps=<int_number>\n"
                     "      Number of keypoints to find in each frame. The default is 1000.\n"
                     "  --ws-local-outlier-rejection=(yes|no)\n"
                     "      Perform local outlier rejection. The default is no.\n\n"
                     "  -sm2=, --save-motions2=(<file_path>|no)\n"
                     "      Save motions estimated for wobble suppression. The default is no.\n"
                     "  -lm2=, --load-motions2=(<file_path>|no)\n"
                     "      Load motions for wobble suppression from file. The default is no.\n\n"
                     "  -gpu=(yes|no)\n"
                     "      Use CUDA optimization whenever possible. The default is no.\n\n"
                     "  -o=, --output=(no|<file_path>)\n"
                     "      Set output file path explicitely. The default is stabilized.avi.\n"
                     "  --fps=(<float_number>|auto)\n"
                     "      Set output video FPS explicitely. By default the source FPS is used (auto).\n"
                     "  --host=(<string>|127.0.0.1)\n"
                     "      Set the UDP host to send video to. Set to 127.0.0.1 by default.\n"
                     "  --port=(<int_number>|5600)\n"
                     "      Set the UDP port to send video to. Set to 5600 by default.\n"
                     //"  -q, --quiet\n"
                     //"      Don't show output video frames.\n\n"
                     "  -h, --help\n"
                     "      Print help.\n\n"
                     "Note: some argument configurations lead to two passes, some to single pass.\n\n";
    }

    class VideoCaptureFrameSrc : public cv::videostab::IFrameSource
    {
    public:
        VideoCaptureFrameSrc(std::string const &pipelineStr)
            : m_pipelineStr{pipelineStr}, m_videoCap{pipelineStr, cv::CAP_GSTREAMER}, m_capFrame{}
        {
        }
        void reset() override
        {
            m_videoCap = cv::VideoCapture(m_pipelineStr, cv::CAP_GSTREAMER);
            m_capFrame = cv::Mat();
        }
        cv::Mat nextFrame() override
        {
            if (!m_videoCap.read(m_capFrame))
            {
                m_capFrame = cv::Mat();
            }
            return m_capFrame;
        }

        cv::VideoCapture &getVideoCapture() { return m_videoCap; }

    private:
        std::string m_pipelineStr;
        cv::VideoCapture m_videoCap;
        cv::Mat m_capFrame;
    };
}

int main(int argc, char **argv)
{
    try
    {
        const char *keys =
            "{ @1                       |           | }"
            "{ m  model                 | affine    | }"
            "{ lp lin-prog-motion-est   | no        | }"
            "{  subset                  | auto      | }"
            "{  thresh                  | auto | }"
            "{  outlier-ratio           | 0.5 | }"
            "{  min-inlier-ratio        | 0.1 | }"
            "{  nkps                    | 1000 | }"
            "{  extra-kps               | 0 | }"
            "{  local-outlier-rejection | no | }"
            "{ sm  save-motions         | no | }"
            "{ lm  load-motions         | no | }"
            "{ r  radius                | 15 | }"
            "{  stdev                   | auto | }"
            "{ lps  lin-prog-stab       | no | }"
            "{  lps-trim-ratio          | auto | }"
            "{  lps-w1                  | 1 | }"
            "{  lps-w2                  | 10 | }"
            "{  lps-w3                  | 100 | }"
            "{  lps-w4                  | 100 | }"
            "{  deblur                  | no | }"
            "{  deblur-sens             | 0.1 | }"
            "{ et  est-trim             | yes | }"
            "{ t  trim-ratio            | 0.1 | }"
            "{ ic  incl-constr          | no | }"
            "{ bm  border-mode          | replicate | }"
            "{  mosaic                  | no | }"
            "{ ms  mosaic-stdev         | 10.0 | }"
            "{ mi  motion-inpaint       | no | }"
            "{  mi-dist-thresh          | 5.0 | }"
            "{ ci color-inpaint         | no | }"
            "{  ci-radius               | 2 | }"
            "{ ws  wobble-suppress      | no | }"
            "{  ws-period               | 30 | }"
            "{  ws-model                | homography | }"
            "{  ws-subset               | auto | }"
            "{  ws-thresh               | auto | }"
            "{  ws-outlier-ratio        | 0.5 | }"
            "{  ws-min-inlier-ratio     | 0.1 | }"
            "{  ws-nkps                 | 1000 | }"
            "{  ws-extra-kps            | 0 | }"
            "{  ws-local-outlier-rejection | no | }"
            "{  ws-lp                   | no | }"
            "{ sm2 save-motions2        | no | }"
            "{ lm2 load-motions2        | no | }"
            "{ gpu                      | no | }"
            "{ o  output                | stabilized.avi | }"
            "{ fps                      | auto | }"
            "{ host                     | 127.0.0.1 | }"
            "{ port                     | 5600 | }"
            //  "{ q quiet                  |  | }"
            "{ h help                   |  | }";
        cv::CommandLineParser cmd(argc, argv, keys);

        // parse command arguments

        if (cmd.get<bool>("help"))
        {
            printHelp();
            return 0;
        }

        if (cmd.get<std::string>("gpu") == "yes")
        {
            std::cout << "initializing GPU...";
            std::cout.flush();
            cv::Mat hostTmp = cv::Mat::zeros(1, 1, CV_32F);
            cv::cuda::GpuMat deviceTmp;
            deviceTmp.upload(hostTmp);
            std::cout << std::endl;
        }
        #if 0
        std::string inputPipeline = "filesrc location=/home/tlyons/Downloads/1000002968.mp4 ! decodebin ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

        #elif 0
        std::string p_line1 = "videotestsrc "; // for PiCam V2
        std::string p_line2 = "! video/x-raw, format=RGBx, width=640, height=480, framerate=200/1 "; // for PiCam V2
        std::string p_line3 = "! videoconvert ! video/x-raw, format=(string)BGR ";
        std::string p_line4 = "! appsink";
        std::string inputPipeline = p_line1 + p_line2 + p_line3 + p_line4;
        #elif 1


        //std::string p_line1 = "libcamerasrc camera-name=/base/axi/pcie@120000/rp1/i2c@88000/imx219@10 "; // for PiCam V2
        std::string p_line1 = "libcamerasrc ";
        // string p_line1 = "libcamerasrc camera-name=/base/axi/pcie@120000/rp1/i2c@80000/imx708@1a ";   // for PiCam V3
        //std::string p_line2 = "! video/x-raw, format=RGBx, width=640, height=480, framerate=200/1 "; // for PiCam V2
        std::string p_line2 = "! capsfilter caps=\"video/x-raw,format=I420,width=1280,height=720,framerate=30/1\" ";
        // string p_line2 = "! video/x-raw, format=RGBx, width=1536, height=864, framerate=120/1 ";  // for PiCam V3
        std::string p_line3 = "! videoconvert ! video/x-raw, format=(string)BGR ";
        std::string p_line4 = "! appsink";
        std::string inputPipeline = p_line1 + p_line2 + p_line3 + p_line4;
        #endif
        std::cout<<"inputPipeline = "<<inputPipeline<<std::endl;
        cv::Ptr<VideoCaptureFrameSrc> frameSource = cv::makePtr<VideoCaptureFrameSrc>(inputPipeline);
        cv::VideoCapture const &videoCap = frameSource->getVideoCapture();
        //cv::Ptr<VideoCaptureFrameSrc> frameSourceOriginal = cv::makePtr<VideoCaptureFrameSrc>(inputPipeline);
        //cv::VideoCapture &videoCapOriginal = frameSourceOriginal->getVideoCapture();
        if (videoCap.isOpened() 
        //&& videoCapOriginal.isOpened()
        )
        {

            cv::videostab::StabilizerBase *stabilizer = nullptr;
            double fps = 0;
            if (cmd.get<std::string>("fps") == "auto")
            {
                fps = videoCap.get(cv::CAP_PROP_FPS);
            }
            else
            {
                fps = cmd.get<double>("fps");
            }
            if (fps <= 0)
            {
                fps = 30;
            }

            cv::Ptr<IMotionEstimatorBuilder> motionEstBuilder;
            if (cmd.get<std::string>("lin-prog-motion-est") == "yes")
                motionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, cmd.get<std::string>("gpu") == "yes"));
            else
                motionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, cmd.get<std::string>("gpu") == "yes"));

            cv::Ptr<IMotionEstimatorBuilder> wsMotionEstBuilder;
            if (cmd.get<std::string>("ws-lp") == "yes")
                wsMotionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, cmd.get<std::string>("gpu") == "yes", "ws-"));
            else
                wsMotionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, cmd.get<std::string>("gpu") == "yes", "ws-"));

            // determine whether we must use one pass or two pass stabilizer
            bool isTwoPass =
                cmd.get<std::string>("est-trim") == "yes" || cmd.get<std::string>("wobble-suppress") == "yes" || cmd.get<std::string>("lin-prog-stab") == "yes";

            cv::Size frameSize((int)videoCap.get(cv::CAP_PROP_FRAME_WIDTH), (int)videoCap.get(cv::CAP_PROP_FRAME_HEIGHT));

            if (isTwoPass)
            {
                // we must use two pass stabilizer

                cv::videostab::TwoPassStabilizer *twoPassStabilizer = new cv::videostab::TwoPassStabilizer();
                stabilizer = twoPassStabilizer;
                twoPassStabilizer->setEstimateTrimRatio(cmd.get<std::string>("est-trim") == "yes");

                // determine stabilization technique

                if (cmd.get<std::string>("lin-prog-stab") == "yes")
                {
                    cv::Ptr<cv::videostab::LpMotionStabilizer> stab = cv::makePtr<cv::videostab::LpMotionStabilizer>();
                    stab->setFrameSize(frameSize);
                    stab->setTrimRatio(cmd.get<std::string>("lps-trim-ratio") == "auto" ? cmd.get<float>("trim-ratio") : cmd.get<float>("lps-trim-ratio"));
                    stab->setWeight1(cmd.get<float>("lps-w1"));
                    stab->setWeight2(cmd.get<float>("lps-w2"));
                    stab->setWeight3(cmd.get<float>("lps-w3"));
                    stab->setWeight4(cmd.get<float>("lps-w4"));
                    twoPassStabilizer->setMotionStabilizer(stab);
                }
                else if (cmd.get<std::string>("stdev") == "auto")
                    twoPassStabilizer->setMotionStabilizer(cv::makePtr<cv::videostab::GaussianMotionFilter>(cmd.get<int>("radius")));
                else
                    twoPassStabilizer->setMotionStabilizer(cv::makePtr<cv::videostab::GaussianMotionFilter>(cmd.get<int>("radius"), cmd.get<float>("stdev")));

                // init wobble suppressor if necessary

                if (cmd.get<std::string>("wobble-suppress") == "yes")
                {
                    cv::Ptr<cv::videostab::MoreAccurateMotionWobbleSuppressorBase> ws = cv::makePtr<cv::videostab::MoreAccurateMotionWobbleSuppressor>();
                    if (cmd.get<std::string>("gpu") == "yes")
#ifdef HAVE_OPENCV_CUDAWARPING
                        ws = makePtr<MoreAccurateMotionWobbleSuppressorGpu>();
#else
                        throw std::runtime_error("OpenCV is built without CUDA support");
#endif

                    ws->setMotionEstimator(wsMotionEstBuilder->build());
                    ws->setPeriod(cmd.get<int>("ws-period"));
                    twoPassStabilizer->setWobbleSuppressor(ws);

                    cv::videostab::MotionModel model = ws->motionEstimator()->motionModel();
                    if (cmd.get<std::string>("load-motions2") != "no")
                    {
                        ws->setMotionEstimator(cv::makePtr<cv::videostab::FromFileMotionReader>(cmd.get<std::string>("load-motions2")));
                        ws->motionEstimator()->setMotionModel(model);
                    }
                    if (cmd.get<std::string>("save-motions2") != "no")
                    {
                        ws->setMotionEstimator(cv::makePtr<cv::videostab::ToFileMotionWriter>(cmd.get<std::string>("save-motions2"), ws->motionEstimator()));
                        ws->motionEstimator()->setMotionModel(model);
                    }
                }
            }
            else
            {
                // we must use one pass stabilizer

                cv::videostab::OnePassStabilizer *onePassStabilizer = new cv::videostab::OnePassStabilizer();
                stabilizer = onePassStabilizer;
                if (cmd.get<std::string>("stdev") == "auto")
                    onePassStabilizer->setMotionFilter(cv::makePtr<cv::videostab::GaussianMotionFilter>(cmd.get<int>("radius")));
                else
                    onePassStabilizer->setMotionFilter(cv::makePtr<cv::videostab::GaussianMotionFilter>(cmd.get<int>("radius"), cmd.get<float>("stdev")));
            }

            stabilizer->setFrameSource(frameSource);
            stabilizer->setMotionEstimator(motionEstBuilder->build());

            // cast stabilizer to simple frame source interface to read stabilized frames
            stabilizedFrames.reset(dynamic_cast<cv::videostab::IFrameSource *>(stabilizer));

            cv::videostab::MotionModel model = stabilizer->motionEstimator()->motionModel();
            if (cmd.get<std::string>("load-motions") != "no")
            {
                stabilizer->setMotionEstimator(cv::makePtr<cv::videostab::FromFileMotionReader>(cmd.get<std::string>("load-motions")));
                stabilizer->motionEstimator()->setMotionModel(model);
            }
            if (cmd.get<std::string>("save-motions") != "no")
            {
                stabilizer->setMotionEstimator(cv::makePtr<cv::videostab::ToFileMotionWriter>(cmd.get<std::string>("save-motions"), stabilizer->motionEstimator()));
                stabilizer->motionEstimator()->setMotionModel(model);
            }

            stabilizer->setRadius(cmd.get<int>("radius"));

            // init deblurer
            if (cmd.get<std::string>("deblur") == "yes")
            {
                cv::Ptr<cv::videostab::WeightingDeblurer> deblurer = cv::makePtr<cv::videostab::WeightingDeblurer>();
                deblurer->setRadius(cmd.get<int>("radius"));
                deblurer->setSensitivity(cmd.get<float>("deblur-sens"));
                stabilizer->setDeblurer(deblurer);
            }

            // set up trimming paramters
            stabilizer->setTrimRatio(cmd.get<float>("trim-ratio"));
            stabilizer->setCorrectionForInclusion(cmd.get<std::string>("incl-constr") == "yes");

            if (cmd.get<std::string>("border-mode") == "reflect")
                stabilizer->setBorderMode(cv::BORDER_REFLECT);
            else if (cmd.get<std::string>("border-mode") == "replicate")
                stabilizer->setBorderMode(cv::BORDER_REPLICATE);
            else if (cmd.get<std::string>("border-mode") == "const")
                stabilizer->setBorderMode(cv::BORDER_CONSTANT);
            else
                throw std::runtime_error("unknown border extrapolation mode: " + cmd.get<std::string>("border-mode"));

            // init inpainter
            cv::videostab::InpaintingPipeline *inpainters = new cv::videostab::InpaintingPipeline();
            cv::Ptr<cv::videostab::InpainterBase> inpainters_(inpainters);
            if (cmd.get<std::string>("mosaic") == "yes")
            {
                cv::Ptr<cv::videostab::ConsistentMosaicInpainter> inp = cv::makePtr<cv::videostab::ConsistentMosaicInpainter>();
                inp->setStdevThresh(cmd.get<float>("mosaic-stdev"));
                inpainters->pushBack(inp);
            }
            if (cmd.get<std::string>("motion-inpaint") == "yes")
            {
                cv::Ptr<cv::videostab::MotionInpainter> inp = cv::makePtr<cv::videostab::MotionInpainter>();
                inp->setDistThreshold(cmd.get<float>("mi-dist-thresh"));
                inpainters->pushBack(inp);
            }
            if (cmd.get<std::string>("color-inpaint") == "average")
                inpainters->pushBack(cv::makePtr<cv::videostab::ColorAverageInpainter>());
            else if (cmd.get<std::string>("color-inpaint") == "ns")
                inpainters->pushBack(cv::makePtr<cv::videostab::ColorInpainter>(int(cv::INPAINT_NS), cmd.get<double>("ci-radius")));
            else if (cmd.get<std::string>("color-inpaint") == "telea")
                inpainters->pushBack(cv::makePtr<cv::videostab::ColorInpainter>(int(cv::INPAINT_TELEA), cmd.get<double>("ci-radius")));
            else if (cmd.get<std::string>("color-inpaint") != "no")
                throw std::runtime_error("unknown color inpainting method: " + cmd.get<std::string>("color-inpaint"));
            if (!inpainters->empty())
            {
                inpainters->setRadius(cmd.get<int>("radius"));
                stabilizer->setInpainter(inpainters_);
            }

            if (cmd.get<std::string>("output") != "no")
                outputPath = cmd.get<std::string>("output");


            std::string host=cmd.get<std::string>("host");
            std::string port=cmd.get<std::string>("port");

            cv::VideoWriter writer{};
            //std::string outputPipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! rtph264pay config-interval=1 pt=96 ! udpsink host=\""+host+"\" port="+port+" sync=false";
            std::string outputPipeline = "appsrc ! videoconvert ! capsfilter caps=\"video/x-raw,format=I420,width=1280,height=720,framerate=30/1\" ! x264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=\""+host+"\" port="+port+" sync=false";

            std::cout<<"outputPipeline = "<<outputPipeline<<std::endl;
            writer.open(outputPipeline, cv::CAP_GSTREAMER, 0, fps, frameSize, true);
            cv::Mat original;
            for (;;)
            {
                
                cv::Mat stabilizedFrame = stabilizedFrames->nextFrame();
                if (stabilizedFrame.empty())
                {
                    break;
                }
                //videoCapOriginal.read(original);
                //cv::imshow("stabilized", stabilizedFrame);
                //cv::imshow("original", original);
                //cv::waitKey(30);
                writer << stabilizedFrame;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "error: " << e.what() << std::endl;
        stabilizedFrames.release();
        return -1;
    }
    stabilizedFrames.release();
    return 0;
}