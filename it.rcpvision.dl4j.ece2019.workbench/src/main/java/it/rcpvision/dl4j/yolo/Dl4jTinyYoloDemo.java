package it.rcpvision.dl4j.yolo;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class Dl4jTinyYoloDemo {

	private static String[] labels = {"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
			"diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};
	
	public static void main(String[] args) throws Exception {
		ComputationGraph model = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
		System.out.println(model.summary()); //Printing the neural network structure
		double dt =  0.4; // Detection threshold
		NativeImageLoader loader = new NativeImageLoader(416, 416, 3);
		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
		Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) model.getOutputLayer(0);

		long t0 = System.currentTimeMillis();
		File imageFile = new File("/home/vincenzo/Pictures/800px-Lex_Av_E_92_St_06.jpg");
		INDArray indArray = loader.asMatrix(imageFile);
		imagePreProcessingScaler.transform(indArray);
		INDArray results = model.outputSingle(indArray);
		List<DetectedObject> detectedObjects = outputLayer.getPredictedObjects(results, dt);
		long t1 = System.currentTimeMillis();
		System.out.println(detectedObjects.size() + " objects detected in "+(t1-t0)+" milliseconds");
		
		drawDetectedObjects(imageFile, detectedObjects); //Drawing detected objects
	}

	private static void drawDetectedObjects(File imageFile, List<DetectedObject> detectedObjects) throws IOException {
		BufferedImage img = ImageIO.read(imageFile);
		Graphics2D g2d = img.createGraphics();
		g2d.setColor(Color.RED);
		g2d.setStroke(new BasicStroke(2));
		for (DetectedObject detectedObject : detectedObjects) {
			double x1 = detectedObject.getTopLeftXY()[0];
			double y1 = detectedObject.getTopLeftXY()[1];
			double x2 = detectedObject.getBottomRightXY()[0];
			double y2 = detectedObject.getBottomRightXY()[1];
			int xs1 = (int) ((x1 / 13.0 ) * (double) img.getWidth());
			int ys1 = (int) ((y1 / 13.0 ) * (double) img.getHeight());
			int xs2 = (int) ((x2 / 13.0 ) * (double) img.getWidth());
			int ys2 = (int) ((y2 / 13.0 ) * (double) img.getHeight());
			g2d.drawString(labels[detectedObject.getPredictedClass()], xs1+4, ys2-2);
			g2d.drawRect(xs1, ys1, xs2-xs1, ys2-ys1);
		}
		JLabel picLabel = new JLabel(new ImageIcon(img));
		JOptionPane.showMessageDialog(null, picLabel, "Image", JOptionPane.PLAIN_MESSAGE, null);
		g2d.dispose();
	}
	
}