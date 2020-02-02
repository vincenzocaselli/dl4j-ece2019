package it.rcpvision.dl4j.workbench;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Step1MnistMLP {

	//The absolute path of the folder containing MNIST training and testing subfolders
	private static final String MNIST_DATASET_ROOT_FOLDER = "/home/vincenzo/dev/dev-dl/data/mnist_png/";
	//Height and widht in pixel of each image
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	//The total number of images into the training and testing set
	private static final int N_SAMPLES_TRAINING = 60000;
	private static final int N_SAMPLES_TESTING = 10000;
	//The number of possible outcomes of the network for each input, 
	//correspondent to the 0..9 digit classification
	private static final int N_OUTCOMES = 10;
	
	//org.deeplearning4j.nn.conf.layers.DenseLayer
    //org.deeplearning4j.nn.weights.WeightInit
	//org.deeplearning4j.nn.conf.layers.OutputLayer
	//org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
	public static void main(String[] args) throws IOException {
		DataSetIterator dsi = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		  .seed(123) //include a random seed for reproducibility
		  // use stochastic gradient descent as an optimization algorithm
		  .updater(new Nesterovs(0.006, 0.9))
		  .l2(1e-4)
		  .list()
		  .layer(new DenseLayer.Builder() //create the first, input layer with xavier initialization
		    .nIn(HEIGHT*WIDTH)
		    .nOut(1000)
		    .activation(Activation.RELU)
		    .weightInit(WeightInit.XAVIER)
		    .build())
		  .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
		    .nIn(1000)
		    .nOut(N_OUTCOMES)
		    .activation(Activation.SOFTMAX)
		    .weightInit(WeightInit.XAVIER)
		    .build())
		  .build();
		
		long t1 = System.currentTimeMillis();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		//print the score with every 500 iteration
		model.setListeners(new ScoreIterationListener(500));
		model.fit(dsi);
		
		
		DataSetIterator testDsi = getDataSetIterator( MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
		Evaluation eval = model.evaluate(testDsi); //org.nd4j.evaluation.classification.Evaluation;
		long t2 = System.currentTimeMillis();
		System.out.println(eval); 
		System.out.println(t2-t1); 
	}

	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
		File folder = new File(folderPath);
		File[] digitFolders = folder.listFiles();

		NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

		INDArray input = Nd4j.create(new int[] { nSamples, HEIGHT * WIDTH });
		INDArray output = Nd4j.create(new int[] { nSamples, N_OUTCOMES });

		int n = 0;
		for (File digitFolder : digitFolders) {
			int labelDigit = Integer.parseInt(digitFolder.getName());
			File[] imageFiles = digitFolder.listFiles();
			for (File imageFile : imageFiles) {
				INDArray img = nil.asRowVector(imageFile);
				scaler.transform(img);
				input.putRow(n, img);
				output.put(n, labelDigit, 1.0);
				n++;
			}
		}
		DataSet dataSet = new DataSet(input, output);
		List<DataSet> listDataSet = dataSet.asList();
		Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
		DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, 10); // batchSize=10
		return dsi;
	}
}
