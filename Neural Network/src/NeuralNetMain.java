
public class NeuralNetMain {
	public static void main(String[] args) {

		NeuralNetwork nn = new NeuralNetwork(2, 2, 1);

		double[][] input1 = { { 0 }, { 1 } };
		double[][] input2 = { { 1 }, { 0 } };
		double[][] input3 = { { 0 }, { 0 } };
		double[][] input4 = { { 1 }, { 1 } };

		double[][] answer1 = { { 1 } };
		double[][] answer2 = { { 1 } };
		double[][] answer3 = { { 0 } };
		double[][] answer4 = { { 0 } };


		// double[][] answer1 = { { 1 }, { 0 } };
		// double[][] answer2 = { { 0 }, { 1 } };
		// double[][] answer3 = { { 1 }, { 1 } };
		// double[][] answer4 = { { 0 }, { 0 } };
		TrainingData[] t = { new TrainingData(new Matrix(input1), new Matrix(answer1)),
				new TrainingData(new Matrix(input2), new Matrix(answer2)),
				new TrainingData(new Matrix(input3), new Matrix(answer3)),
				new TrainingData(new Matrix(input4), new Matrix(answer4)) };

		for (int i = 0; i < 0; i++) {
			TrainingData training = t[(int) (Math.random() * 4)];
			nn.train(training.input, training.answer);
		}

		nn.feedFoward(new Matrix(input1)).show();
		nn.feedFoward(new Matrix(input2)).show();
		nn.feedFoward(new Matrix(input3)).show();
		nn.feedFoward(new Matrix(input4)).show();

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
	int inputNodes, hiddenNodes, outputNodes;
	Matrix ihWs, hoWs, hBs, oBs;
	double learningRate = .1;

	public NeuralNetwork(int is, int hs, int os) {
		inputNodes = is;
		hiddenNodes = hs;
		outputNodes = os;
		ihWs = Matrix.random(hiddenNodes, inputNodes);
		hoWs = Matrix.random(outputNodes, hiddenNodes);

		hBs = Matrix.random(hiddenNodes, 1);
		oBs = Matrix.random(outputNodes, 1);

	}

	public Matrix feedFoward(Matrix input) {
		Matrix hidden = ihWs.timesM(input).plus(hBs);
		hidden = hidden.mapSigmoid();

		Matrix output = hoWs.timesM(hidden).plus(oBs);
		output = output.mapSigmoid();

		return output;
	}

	public void train(Matrix inputs, Matrix targets) {

		// Feed forward Section:
		Matrix hidden = ihWs.timesM(inputs);
		hidden = hidden.plus(hBs).mapSigmoid();

		// hidden has bias added and is S-mapped

		Matrix outputs = hoWs.timesM(hidden);
		outputs = outputs.plus(oBs).mapSigmoid();

		Matrix outputErrors = targets.minus(outputs);
		//outputErrors.show();

		Matrix gradients = outputs.mapDsigmoid();

		gradients.times(outputErrors); // Error here
		gradients.scalarMult(learningRate);

		// Matrix hiddenT = hidden.transpose();
		Matrix hoDeltaW = gradients.timesM(hidden.transpose());

		// adjust output weights and bias's
		this.hoWs = hoWs.plus(hoDeltaW);
		this.oBs = oBs.plus(gradients);

		// hidden layer error calculation
		// Matrix hoWsT = hoWs.transpose();
		Matrix hiddenErrors = hoWs.transpose().timesM(outputErrors);

		// calc hidden gradient
		Matrix hiddenGradient = hidden; // transposed this as an edit...
		hiddenGradient.mapDsigmoid();

		hiddenGradient.times(hiddenErrors);
		// TODO Problem : hiddenGradient dimensions not matching hE dims
		hiddenGradient.scalarMult(learningRate);

		// calc hidden deltas
		Matrix inputsT = inputs.transpose();
		Matrix ihDeltaW = hiddenGradient.timesM(inputsT);

		this.ihWs = ihWs.plus(ihDeltaW);
		this.hBs = hBs.plus(hiddenGradient);

		// outputs.show();
		// targets.show();
		// outputErrors.show();

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