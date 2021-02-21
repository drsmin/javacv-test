package kr.hubble.javacv.test;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class CropPhoto {
	public static void main(String[] args) throws IOException, URISyntaxException, InterruptedException {
		// Load Face Detector
		URL url = new URL(
				"https://raw.github.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml");
		File file = Loader.cacheResource(url);
		String classifierName = file.getAbsolutePath();

		CascadeClassifier faceDetector = new CascadeClassifier(classifierName);

		String trainingDir = "C:\\Users\\drs\\Pictures";

		File root = new File(trainingDir);

		FilenameFilter imgFilter = new FilenameFilter() {
			public boolean accept(File dir, String name) {
				name = name.toLowerCase();
				return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
			}
		};

		File[] imageFiles = root.listFiles(imgFilter);

		for (File image : imageFiles) {

			System.out.println(image.getAbsolutePath());

			// Load image
			Mat img = imread(image.getAbsolutePath());
			
			System.out.println(img);

			// Find faces on the image
			RectVector faces = new RectVector();
			faceDetector.detectMultiScale(img, faces);

			System.out.println("Faces detected: " + faces.size());

			Rect rect = faces.get(0);

			Rect rectCrop = new Rect(rect.x(), rect.y(), rect.width(), rect.height());
			Mat croppedImage = new Mat(img, rectCrop);

			// for (Rect rect : faces.get() ) {
			// Core.rectangle(img, new Point(rect.x(), rect.y()), new Point(rect.x() +
			// rect.width(), rect.y() + rect.height()),
			// new Scalar(0, 255, 0));
			// }

			// Save results
			imwrite(image.getAbsolutePath(), croppedImage);

		}
	}
}