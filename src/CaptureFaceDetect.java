import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class CaptureFaceDetect {
    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        Mat grayFrame = new Mat();
        Mat grayHistoEqFrame = new Mat();
        Mat dst = new Mat();

        String faceCascade = "./models/haarcascades/haarcascade_frontalface_alt.xml";

        CascadeClassifier faceClassifier = new CascadeClassifier();
        faceClassifier.load(faceCascade);

        VideoCapture cam = new VideoCapture(0);

        if(!cam.isOpened()) return;

        while (true){
            cam.read(src);
            if(src.empty()) break;

            Imgproc.cvtColor(src, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayHistoEqFrame);

            MatOfRect faces = new MatOfRect();
            faceClassifier.detectMultiScale(grayFrame,faces);

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(src, facesArray[i], new Scalar(0,128,0),2);
            }

            HighGui.imshow("Original",src);
            HighGui.imshow("Gray",grayFrame);
            HighGui.imshow("HistoEq",grayHistoEqFrame);


            int key = HighGui.waitKey(20);
            if(key == 27)
                break;
        }
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
