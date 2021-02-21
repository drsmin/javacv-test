package kr.hubble.javacv.test;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.net.URL;
import java.nio.IntBuffer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class PredictPhoto {

	public static void main(String[] args) throws IOException {
		
		URL url = new URL("https://raw.github.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml");
        File file = Loader.cacheResource(url);
        String classifierName = file.getAbsolutePath();
    	
        CascadeClassifier faceDetector = new CascadeClassifier (classifierName);
		
		//String trainingDir = args[0];
        Mat img = imread("C:\\Users\\drs\\face.jpg", IMREAD_GRAYSCALE);

        System.out.println(img.toString());
        
        // Find faces on the image
        RectVector faces = new RectVector ();
        faceDetector.detectMultiScale(img, faces);
        
        Rect rect = faces.get(0);
        
        Rect rectCrop = new Rect(rect.x(), rect.y(), rect.width(), rect.height());
        Mat croppedImage = new Mat(img, rectCrop);
        
        imwrite("C:\\Users\\drs\\face2.jpg", croppedImage);

        String trainingDir = "C:\\Users\\drs\\Pictures";
        
        File root = new File(trainingDir);
        
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;
        
        for (File image : imageFiles) {
        	
        	System.out.println(image.getAbsoluteFile().getName());
        	
            Mat imgT = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);

            int label = Integer.parseInt(image.getName().split("\\-")[0]);

            images.put(counter, imgT);

            labelsBuf.put(counter, label);

            counter++;
        }

        // FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
        // FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(images, labels);

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(0);
        
        faceRecognizer.predict(croppedImage, label, confidence);
        
        int predictedLabel = label.get(0);

        System.out.println("Predicted label: " + predictedLabel);
        System.out.println("confidence " + confidence.get(0));
        
    }

}
