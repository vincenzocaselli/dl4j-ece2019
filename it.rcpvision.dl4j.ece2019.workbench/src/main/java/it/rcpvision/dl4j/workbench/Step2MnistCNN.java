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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Step2MnistCNN {

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
	
	public static void main(String[] args) throws IOException {
		long t0 = System.currentTimeMillis();
		DataSetIterator dsi = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);
		
		int channels = 1;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(0.006, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .nIn(channels )
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(N_OUTCOMES)
                    .activation(Activation.SOFTMAX)
                    .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, channels)) // InputType.convolutional for normal image
                .build();
		
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		//print the score with every 500 iteration
		model.setListeners(new ScoreIterationListener(500));
		model.fit(dsi);
		
		
		DataSetIterator testDsi = getDataSetIterator( MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
		Evaluation eval = model.evaluate(testDsi); //org.nd4j.evaluation.classification.Evaluation;
		System.out.println(eval); 
		
		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
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
