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
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Step3MalariaImagesClassifier {
	//Height and widht in pixel of each image
	private static final int HEIGHT = 100;
	private static final int WIDTH = 100;
	//The total number of images into the training and testing set
	private static final int N_SAMPLES_TRAINING = 13779+13779;
//	private static final int N_SAMPLES_TESTING = 10000;
	//The number of possible outcomes of the network for each input, 
	//correspondent to the 0..9 digit classification
	private static final int N_OUTCOMES = 2;

	private static Logger log = LoggerFactory.getLogger(Step3MalariaImagesClassifier.class);

	private static String datasetRootFolder;
		
	public static void main(String[] args) throws IOException {

		//The absolute path of the folder containing MNIST training and testing subfolders
		String sys = System.getProperty("os.name");
		if (sys.contains("Linux")) {
			datasetRootFolder = "/DISK-G/ML/detect-malaria/";
		} else {
			datasetRootFolder = "G:/ML/detect-malaria/";
		}
		
		long t0 = System.currentTimeMillis();

		DataSetIterator dsi = getDataSetIterator(datasetRootFolder, N_SAMPLES_TRAINING);

		int rngSeed = 123;
        int nEpochs = 2; // Number of training epochs
        
        log.info("Build model....");
        
//        Map<Integer, Double> learningRateSchedule = new HashMap<>();
//        learningRateSchedule.put(0, 0.06);
//        learningRateSchedule.put(200, 0.05);
//        learningRateSchedule.put(600, 0.028);
//        learningRateSchedule.put(800, 0.0060);
//        learningRateSchedule.put(1000, 0.001);
        
        int channels = 3;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs( 0.005, 0.9
                		//new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)
                		))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels )
                    .stride(1, 1)
                    .nOut(40)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(5, 5)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(5, 5)
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
        model.setListeners(new ScoreIterationListener(1));
        log.info("Train model....");
        model.fit(dsi, nEpochs);

		DataSetIterator testDsi = getDataSetIterator( datasetRootFolder , N_SAMPLES_TRAINING);
        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(testDsi);
        log.info(eval.stats());

		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		log.info("\n\nTotal time: "+t+" seconds");
	}

	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
		File folder = new File(folderPath);
		File[] digitFolders = folder.listFiles();
		
		NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH, 3);
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);

		INDArray input = Nd4j.create(new int[]{ nSamples, 3, HEIGHT, WIDTH });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });

		int n = 0;
		//scan all 0..9 digit subfolders
		for (File digitFolder : digitFolders) {
			//take note of the digit in processing, since it will be used as a label
			int labelDigit = digitFolder.getName().equalsIgnoreCase("Parasitized") ? 1 : 0 ;
			//scan all the images of the digit in processing
			File[] imageFiles = digitFolder.listFiles();
			for (File imageFile : imageFiles) {
				//read the image as a one dimensional array of 0..255 values
//				INDArray img = nil.asRowVector(imageFile);
				INDArray img = nil.asMatrix(imageFile);
				log.info(img.shapeInfoToString());
				//scale the 0..255 integer values into a 0..1 floating range
				//Note that the transform() method returns void, since it updates its input array
				scaler.transform(img);
				//copy the img array into the input matrix, in the next row
				input.putRow( n, img );
				//in the same row of the output matrix, fire (set to 1 value) the column correspondent to the label
				output.put( n, labelDigit, 1.0 );
				//row counter increment
				n++;
				log.info(labelDigit+" \t "+n);
			}
		}

		//Join input and output matrixes into a dataset
		DataSet dataSet = new DataSet( input, output );
		//Convert the dataset into a list
		List<DataSet> listDataSet = dataSet.asList();
		//Shuffle its content randomly
		Collections.shuffle( listDataSet, new Random(System.currentTimeMillis()) );
		//Set a batch size
		int batchSize = 54;
		//Build and return a dataset iterator that the network can use
		DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, batchSize );
		return dsi;
	}

}
