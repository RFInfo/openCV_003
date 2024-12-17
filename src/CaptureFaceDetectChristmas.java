import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.List;


public class CaptureFaceDetectChristmas {

	public static void main(String[] args) {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat logo = Imgcodecs.imread("./test_images/santa-claus-hat.png", Imgcodecs.IMREAD_UNCHANGED);
		Mat logoMask = new Mat();
		Core.extractChannel(logo,logoMask, 3);
		Mat logoBGR = new Mat();
		Imgproc.cvtColor(logo, logoBGR, Imgproc.COLOR_BGRA2BGR);
		Mat logoResized = new Mat();

		String filenameFaceCascade = "./models/haarcascades/haarcascade_frontalface_alt.xml";

		int cameraDevice = 0;

		CascadeClassifier faceCascade = new CascadeClassifier();

		if (!faceCascade.load(filenameFaceCascade)) {
			System.err.println("--(!)Error loading face cascade: " + filenameFaceCascade);
			System.exit(1);
		}


		VideoCapture capture = new VideoCapture(cameraDevice);
//		VideoCapture capture = new VideoCapture("./test_data/head-pose-face-detection-female-and-male.mp4");
//		capture.open(2);
		
//		capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
//		capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);
//		capture.set(Videoio.CAP_PROP_FPS, 1);
//		capture.set(Videoio.CAP_FFMPEG, 1);
		
		System.out.println(capture.get(Videoio.CAP_PROP_FRAME_WIDTH) + " " + capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
		System.out.println(capture.get(Videoio.CAP_PROP_FPS));
		System.out.println(capture.get(Videoio.CAP_PROP_BUFFERSIZE));
		
		if (!capture.isOpened()) {
			System.err.println("--(!)Error opening video capture");
			System.exit(0);
		}

		Mat frame = new Mat();
		while (capture.read(frame)) {
			if (frame.empty()) {
				System.err.println("--(!) No captured frame -- Break!");
				break;
			}
			
			// preprocess
			Mat frameGray = new Mat();
			Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
			Imgproc.equalizeHist(frameGray, frameGray);

			// detect faces
			MatOfRect faces = new MatOfRect();
			faceCascade.detectMultiScale(frameGray, faces);

			List<Rect> listOfFaces = faces.toList();
			for (Rect face : listOfFaces) {
				Point center = new Point(face.x + face.width / 2, face.y + face.height / 2);
				Imgproc.ellipse(frame, center, new Size(face.width / 2, face.height / 2), 0, 0, 360,
						new Scalar(255, 255, 0));

				Imgproc.rectangle(frame, face, new Scalar(255, 0, 0));

				Mat faceROI = frameGray.submat(face);
				
				int x = face.x;
				int y = face.y-face.height;
				if(y <= 0)
					y = 0;
				Rect subRect = new Rect(x,y,face.width,face.height);
				Mat subROI = frame.submat(subRect);
			
				// prepare hat
				Imgproc.resize(logoBGR, logoResized, faceROI.size(),0,0,Imgproc.INTER_LINEAR);
				Mat logoMaskResized = new Mat();
				Imgproc.resize(logoMask, logoMaskResized, faceROI.size());

				// add hat
				Core.copyTo(logoResized,subROI, logoMaskResized);
			}

			HighGui.imshow("Santa", frame);

			if (HighGui.waitKey(25) == 27) {
				break;// escape
			}
		}
//		capture.release();
		HighGui.destroyAllWindows();
		System.exit(0);
	}
}