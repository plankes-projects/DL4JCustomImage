package DL4JCustomImage.DL4JCustomImage;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.canova.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
	private static File IMAGE_ROOT = new File("images");
	private static int IMAGE_WIDTH = 100;
	private static int IMAGE_HEIGHT = 100;
	private static int CHANNELS = 3;

	private static List<String> LABELS = Arrays.asList("duck", "dog");
	private static INDArray DUCK_LABEL = Nd4j.create(new float[] { 1, 0 });
	private static INDArray DOG_LABEL = Nd4j.create(new float[] { 0, 1 });
	private static int DUCK_INDEX = 0;
	private static int DOG_INDEX = 1;

	private static File DUCK_IMAGE = new File(IMAGE_ROOT, "duck.jpg");
	private static File DOG_IMAGE = new File(IMAGE_ROOT, "dog.jpg");

	private static File DUCK_TEST_IMAGE = new File(IMAGE_ROOT, "duck_test.jpg");
	private static File DOG_TEST_IMAGE = new File(IMAGE_ROOT, "dog_test.jpg");

	/**
	 * put -Djava.library.path="" into run enviroments
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		int numLabels = LABELS.size();
		System.out.println(IMAGE_ROOT.getAbsolutePath());

		// setup model
		MultiLayerNetwork model = new MultiLayerNetwork(buildConfig(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, numLabels));
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		// train model
		int epochs = 10;
		for (int i = 0; i < epochs; i++) {
			System.out.println("Epoch: " + i);
			DataSet ds1 = new DataSet(featureFromImage(DUCK_IMAGE), DUCK_LABEL);
			DataSet ds2 = new DataSet(featureFromImage(DOG_IMAGE), DOG_LABEL);
			DataSet dsMerge = DataSet.merge(Arrays.asList(ds1, ds2));
			model.fit(dsMerge);
		}

		// test model
		Evaluation eval = new Evaluation(LABELS);
		eval.eval(DUCK_LABEL, model.output(featureFromImage(DUCK_IMAGE), false));
		eval.eval(DOG_LABEL, model.output(featureFromImage(DOG_IMAGE), false));
		System.out.println(eval.stats());

		// predict new image
		testFile(model, DOG_TEST_IMAGE);
		testFile(model, DUCK_TEST_IMAGE);

	}

	static private void testFile(MultiLayerNetwork model, File f) throws Exception {
		INDArray data = featureFromImage(f);
		INDArray prediction = model.output(data, false);

		if (prediction.getFloat(DUCK_INDEX) > prediction.getFloat(DOG_INDEX)) {
			System.out.println("I think " + f.getName() + " is a duck");
		} else {
			System.out.println("I think " + f.getName() + " is a dog");
		}
	}

	static private MultiLayerConfiguration buildConfig(int imageWidth, int imageHeight, int channels, int numOfClasses) {
		int seed = 123;
		int iterations = 1;

		WeightInit weightInit = WeightInit.XAVIER;
		String activation = "relu";
		Updater updater = Updater.NESTEROVS;
		double lr = 1e-3;
		double mu = 0.9;
		double l2 = 5e-4;
		boolean regularization = true;

		SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;
		double nonZeroBias = 1;
		double dropOut = 0.5;

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations).activation(activation).weightInit(weightInit)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(lr).momentum(mu)
				.regularization(regularization).l2(l2).updater(updater).useDropConnect(true)

				// AlexNet
				.list().layer(0, new ConvolutionLayer.Builder(new int[] { 11, 11 }, new int[] { 4, 4 }, new int[] { 3, 3 }).name("cnn1").nIn(channels).nOut(96).build())
				.layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
				.layer(2, new SubsamplingLayer.Builder(poolingType, new int[] { 3, 3 }, new int[] { 2, 2 }).name("maxpool1").build())
				.layer(3, new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 1, 1 }, new int[] { 2, 2 }).name("cnn2").nOut(256).biasInit(nonZeroBias).build())
				.layer(4, new LocalResponseNormalization.Builder().name("lrn2").k(2).n(5).alpha(1e-4).beta(0.75).build())
				.layer(5, new SubsamplingLayer.Builder(poolingType, new int[] { 3, 3 }, new int[] { 2, 2 }).name("maxpool2").build())
				.layer(6, new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name("cnn3").nOut(384).build())
				.layer(7, new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name("cnn4").nOut(384).biasInit(nonZeroBias).build())
				.layer(8, new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name("cnn5").nOut(256).biasInit(nonZeroBias).build())
				.layer(9, new SubsamplingLayer.Builder(poolingType, new int[] { 3, 3 }, new int[] { 2, 2 }).name("maxpool3").build())
				.layer(10, new DenseLayer.Builder().name("ffn1").nOut(4096).biasInit(nonZeroBias).dropOut(dropOut).build())
				.layer(11, new DenseLayer.Builder().name("ffn2").nOut(4096).biasInit(nonZeroBias).dropOut(dropOut).build())
				.layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numOfClasses).activation("softmax").build()).backprop(true).pretrain(false)
				.cnnInputSize(imageHeight, imageWidth, channels);

		return builder.build();
	}

	static public INDArray featureFromImage(@Nonnull File image) throws Exception {
		NativeImageLoader imageLoader = new NativeImageLoader(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS);
		INDArray vector = imageLoader.asRowVector(image);
		// vector = normalize(vector);
		return vector;
	}
}
