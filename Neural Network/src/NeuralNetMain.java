import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetMain {
	public static void main(String[] args) {

		NeuralNetwork nn = new NeuralNetwork(2,2,1);
		double[][] input7 = { { 0 }, { 0 } };
		System.out.println((nn.feedFoward(new Matrix(input7)).data[0][0]));

		double[][] input1 = { { 0 }, { 1 } };
		double[][] input2 = { { 1 }, { 0 } };
		double[][] input3 = { { 0 }, { 0 } };
		double[][] input4 = { { 1 }, { 1 } };
		
		double[][] answer1 = { { 1 } };
		double[][] answer2 = { { 1 } };
		double[][] answer3 = { { 0 } };
		double[][] answer4 = { { 0 } };
		
		//TrainingData[] t = {new TrainingData(new Matrix(input1), new Matrix(answer1))
		nn.train(new Matrix(input1), new Matrix(answer1));
		
		ArrayList<TrainingData> train = new ArrayList<TrainingData>();
		for (int i = 0; i < 1000; i++) {
			double i1 = Math.random()*.5;
			double ans = i1*2;

			double[][] input = { { i1 } };
			double[][] answer = { { ans } };
			train.add(new TrainingData(new Matrix(input), new Matrix(answer)));
		}


		// double[][] answer1 = { { 1 }, { 0 } };
		// double[][] answer2 = { { 0 }, { 1 } };
		// double[][] answer3 = { { 1 }, { 1 } };
		// double[][] answer4 = { { 0 }, { 0 } };
		


	}

	public static double sig(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static double iSig(double x) {
		return -Math.log(1 / x - 1);
	}
}

class TrainingData {
	Matrix input, answer;

	public TrainingData(Matrix input, Matrix answer) {
		super();
		this.input = input;
		this.answer = answer;
	}

}

class NeuralNetwork {
	int[] nodes;
	Matrix[] weights;
	Matrix[] biases;
	Matrix[] layers;
	double learningRate = .01;

	public NeuralNetwork(int... nodes) {
		this.nodes = nodes;
		layers = new Matrix[nodes.length];
		weights = new Matrix[nodes.length-1];
		biases = new Matrix[nodes.length-1];
		for(int i = 0; i<weights.length; i++) {
			weights[i] = Matrix.random(nodes[i+1], nodes[i]);
			biases[i] = Matrix.random(nodes[i+1], 1);
		}

	}

	public Matrix feedFoward(Matrix input) {
		layers[0] = input;
		for(int i = 0; i<nodes.length-1; i++) {
		layers[i + 1] = weights[i].timesM(layers[i]).plus(biases[i]).mapSigmoid();
		}
		return layers[layers.length-1];
	}

	public void train(Matrix inputs, Matrix targets) {

		feedFoward(inputs); //now we have all the layers stuff
		
		//start back propagating
		Matrix[] errors = new Matrix[nodes.length-1];
		
		
		errors[0] = targets.minus(layers[layers.length - 1]); //targets - outputs = error
		for(int i = 0; i<nodes.length-1; i++) { //back propagates through the layers
			if(i != 0) {
				//new error = prev weights transposed times prev errors
				errors[i] = weights[nodes.length - 1 - i].transpose().timesM(errors[i-1]); 
			}
			Matrix gradient = layers[nodes.length-1-i].mapDsigmoid();
			gradient.times(errors[i]); //added -1 ?
			gradient.scalarMult(learningRate);
			Matrix deltaWs = gradient.timesM(layers[nodes.length-2-i].transpose());
			weights[weights.length - 1 - i] = weights[weights.length - 1 - i].plus(deltaWs);
			biases[biases.length - 1 - i] = biases[biases.length - 1 - i].plus(gradient);
		}


	}
}

class Matrix {
	int M; // number of rows
	int N; // number of columns
	double[][] data; // M-by-N array

	// create M-by-N matrix of 0's
	public Matrix(int M, int N) {
		this.M = M;
		this.N = N;
		data = new double[M][N];
	}

	// create matrix based on 2d array
	public Matrix(double[][] data) {
		M = data.length;
		N = data[0].length;
		this.data = new double[M][N];
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				this.data[i][j] = data[i][j];
	}

	// copy constructor
	private Matrix(Matrix A) {
		this(A.data);
	}

	// create and return a random M-by-N matrix with values between -1 and 1
	public static Matrix random(int M, int N) {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				A.data[i][j] = Math.random() * 2 - 1;
		return A;
	}

	// create and return the N-by-N identity matrix
	public static Matrix identity(int N) {
		Matrix I = new Matrix(N, N);
		for (int i = 0; i < N; i++)
			I.data[i][i] = 1;
		return I;
	}

	// swap rows i and j
	private void swap(int i, int j) {
		double[] temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

	// create and return the transpose of the invoking matrix
	public Matrix transpose() {
		Matrix A = new Matrix(N, M);
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				A.data[j][i] = this.data[i][j];
		return A;
	}

	// return C = A + B
	public Matrix plus(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N)
			throw new RuntimeException("Illegal matrix dimensions.");
		Matrix C = new Matrix(M, N);
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				C.data[i][j] = A.data[i][j] + B.data[i][j];
		return C;
	}

	// return C = A - B
	public Matrix minus(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N)
			throw new RuntimeException("Illegal matrix dimensions.");
		Matrix C = new Matrix(M, N);
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				C.data[i][j] = A.data[i][j] - B.data[i][j];
		return C;
	}

	// does A = B exactly?
	public boolean eq(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N)
			throw new RuntimeException("Illegal matrix dimensions.");
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				if (A.data[i][j] != B.data[i][j])
					return false;
		return true;
	}

	// return C = A * B
	public void times(Matrix B) {

		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				data[i][j] *= B.data[i][j];
			}
		}
	}

	public Matrix timesM(Matrix B) {

		Matrix A = this;
		if (A.N != B.M)
			throw new RuntimeException("Illegal matrix dimensions.");
		Matrix C = new Matrix(A.M, B.N);
		for (int i = 0; i < C.M; i++)
			for (int j = 0; j < C.N; j++)
				for (int k = 0; k < A.N; k++)
					C.data[i][j] += (A.data[i][k] * B.data[k][j]);
		return C;
	}

	// return x = A^-1 b, assuming A is square and has full rank
	public Matrix solve(Matrix rhs) {
		if (M != N || rhs.M != N || rhs.N != 1)
			throw new RuntimeException("Illegal matrix dimensions.");

		// create copies of the data
		Matrix A = new Matrix(this);
		Matrix b = new Matrix(rhs);

		// Gaussian elimination with partial pivoting
		for (int i = 0; i < N; i++) {

			// find pivot row and swap
			int max = i;
			for (int j = i + 1; j < N; j++)
				if (Math.abs(A.data[j][i]) > Math.abs(A.data[max][i]))
					max = j;
			A.swap(i, max);
			b.swap(i, max);

			// singular
			if (A.data[i][i] == 0.0)
				throw new RuntimeException("Matrix is singular.");

			// pivot within b
			for (int j = i + 1; j < N; j++)
				b.data[j][0] -= b.data[i][0] * A.data[j][i] / A.data[i][i];

			// pivot within A
			for (int j = i + 1; j < N; j++) {
				double m = A.data[j][i] / A.data[i][i];
				for (int k = i + 1; k < N; k++) {
					A.data[j][k] -= A.data[i][k] * m;
				}
				A.data[j][i] = 0.0;
			}
		}

		// back substitution
		Matrix x = new Matrix(N, 1);
		for (int j = N - 1; j >= 0; j--) {
			double t = 0.0;
			for (int k = j + 1; k < N; k++)
				t += A.data[j][k] * x.data[k][0];
			x.data[j][0] = (b.data[j][0] - t) / A.data[j][j];
		}
		return x;

	}

	public void scalarMult(double d) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				this.data[i][j] *= d;
			}
		}

	}

	// print matrix to standard output
	public void show() {

		for (int i = 0; i < M; i++) {

			for (int j = 0; j < N; j++) {
				System.out.print(this.data[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}

	public Matrix mapSigmoid() {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[i][j] = sigmoid(this.data[i][j]);
			}
		}
		return A;
	}

	public Matrix mapDsigmoid() {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[i][j] = dsigmoid(this.data[i][j]);
			}
		}
		return A;
	}

	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public double dsigmoid(double x) {
		return x * (1 - x);
	}

	public void showDimensions() {
		System.out.println(this.M + "x" + this.N);
	}
}