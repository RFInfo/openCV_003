import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class CaptureFaceDetectHalloween {


    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String face_cascade_file = "./models/haarcascades/haarcascade_frontalface_alt.xml";

        CascadeClassifier face_cascade = new CascadeClassifier();
        face_cascade.load(face_cascade_file);

        Mat src = new Mat();
        Mat dst = new Mat();
        Mat gray = new Mat();

        Mat logo = Imgcodecs.imread("./test_images/halloween.png", Imgcodecs.IMREAD_UNCHANGED);
        Mat logoMask = new Mat();
        Core.extractChannel(logo, logoMask, 3);
        Imgproc.cvtColor(logo, logo, Imgproc.COLOR_BGRA2BGR);
        Mat logoResized = new Mat();
        Mat logoMaskResized = new Mat();
//        src = Imgcodecs.imread("./test_data/workplace-1245776_1280.jpg");

//        VideoCapture videoCapture = new VideoCapture("./test_data/video.mp4");
        VideoCapture videoCapture = new VideoCapture(0);

//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);
//        videoCapture.set(Videoio.CAP_PROP_FPS,8);

        if (!videoCapture.isOpened()) return;

        Rect[] facesArray = new Rect[0];
        int frameCounter = 0;

        while (true) {
            if (!videoCapture.read(src)) break;
            frameCounter++;

//                Imgproc.resize(src,src,new Size(),0.30,0.30);
//                dst = src.clone();
            src.copyTo(dst);

            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(gray, gray);

            // process each 5th frame
            if (frameCounter % 5 == 0) {
                MatOfRect faces = new MatOfRect();
//              face_cascade.detectMultiScale(gray,faces);
                face_cascade.detectMultiScale(gray, faces, 1.1, 4, Objdetect.CASCADE_SCALE_IMAGE);
                facesArray = faces.toArray();
            }

            for (int i = 0; i < facesArray.length; i++) {
//                    System.out.println(facesArray[i]);
                Mat faceROI = dst.submat(facesArray[i]);

                Imgproc.resize(logo, logoResized, new Size(faceROI.width(), faceROI.height()));
                Imgproc.resize(logoMask, logoMaskResized, new Size(faceROI.width(), faceROI.height()));

                Core.copyTo(logoResized, faceROI, logoMaskResized);

                Imgproc.rectangle(dst, facesArray[i], new Scalar(0, 255, 0), 2);
            }

            HighGui.imshow("Dst", dst);
            int key = HighGui.waitKey(10);
            if (key == 27)
                break;
        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
