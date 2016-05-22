package LinearRegression;

import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Simple Linear Regression implementation
 */
public class LinearRegression {
    public static void linearRegression() throws Exception {
        Matrix trainingData = MatrixData.getDataMatrix("data/linear_regression/linear-regression-train.csv");
        // getMatrix(Initial row index, Final row index, Initial column index, Final column index)
        Matrix train_x = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, 0, trainingData.getColumnDimension() - 2);
        Matrix train_y = trainingData.getMatrix(0, trainingData.getRowDimension()-1, trainingData.getColumnDimension()-1, trainingData.getColumnDimension()-1);

        Matrix testData = MatrixData.getDataMatrix("data/linear_regression/linear-regression-test.csv");
        Matrix test_x = testData.getMatrix(0, testData.getRowDimension() - 1, 0, testData.getColumnDimension() - 2);
        Matrix test_y = testData.getMatrix(0, testData.getRowDimension() - 1, testData.getColumnDimension() - 1,testData.getColumnDimension() - 1);

        /* Linear Regression */
        /* 2 step process */
        // 1) find beta
        
        Matrix new_train_x = addMatrixColumnWith1(train_x);
        Matrix new_test_x = addMatrixColumnWith1(test_x);
        
        Matrix beta = getBeta(new_train_x, train_y);
        Matrix stoch_Beta = getBetaforStochastic_Gradient(new_train_x, train_y);
        Matrix batch_Beta = getBetaforBatch_Gradient(new_train_x, train_y);
        // 2) predict y for test data using beta calculated from train data
                
        Matrix predictedY = new_test_x.times(beta);
        Matrix stoch_predictedY = new_test_x.times(stoch_Beta);
        Matrix batch_predictedY = new_test_x.times(batch_Beta);
        
        // Output
        printOutput(predictedY, "Linear-regression-output--Closed_Form");
        printOutput(beta, "Linear-regression-Beta-output--Closed_Form");
        printOutput(stoch_Beta, "Linear-regression-Beta-output--Stochastic_Gradient");
        printOutput(stoch_predictedY, "Linear-regression-output--Stochastic_Gradient");
        printOutput(batch_Beta, "Linear-regression-Beta-output--Batch_Gradient");
        printOutput(batch_predictedY, "Linear-regression-output--Batch_Gradient");
        
        
        
        double mse = getMeanSquareError(test_y, predictedY);
        double stoch_mse = getMeanSquareError(test_y, stoch_predictedY);
        double batch_mse = getMeanSquareError(test_y, batch_predictedY);
        //System.out.println(batch_mse);
        printMSE(mse, "Linear-regression-MSE--Closed_Form");
        printMSE(stoch_mse, "Linear-regression-MSE--Stochastic_Gradient");
        printMSE(batch_mse, "Linear-regression-MSE--Batch_Gradient");
        
        Matrix ztrain_x = zNormalize(train_x);
        ztrain_x = addMatrixColumnWith1(ztrain_x);
        
        Matrix ztrain_y = zNormalize(train_y);
        
        beta = getBeta(ztrain_x, ztrain_y);
        stoch_Beta = getBetaforStochastic_Gradient(ztrain_x, ztrain_y);
        batch_Beta = getBetaforBatch_Gradient(ztrain_x, ztrain_y);
        
        Matrix znew_test_x = zNormalize(new_test_x);
        predictedY = znew_test_x.times(beta);
        stoch_predictedY = znew_test_x.times(stoch_Beta); 
        batch_predictedY = znew_test_x.times(batch_Beta);
        
        
        printOutput(predictedY, "Linear-regression-output_with_z_Normalization--Closed_Form");
        printOutput(beta, "Linear-regression-Beta-output-with_z_Normalization--Closed_Form");
        printOutput(stoch_predictedY, "Linear-regression-output_with_z_Normalization--Stochastic_Gradient");
        printOutput(stoch_Beta, "Linear-regression-Beta-output-with_z_Normalization--Stochastic_Gradient");
        printOutput(batch_predictedY, "Linear-regression-output_with_z_Normalization--Batch_Gradient");
        printOutput(batch_Beta, "Linear-regression-Beta-output-with_z_Normalization--Batch_Gradient");
        
        
        Matrix znew_test_y = zNormalize(test_y);
        mse = getMeanSquareError(znew_test_y, predictedY);
        stoch_mse = getMeanSquareError(znew_test_y, stoch_predictedY);
        batch_mse = getMeanSquareError(znew_test_y, batch_predictedY);
        printMSE(mse, "Linear-regression-MSE_with_Z_Normalization--Closed_Form");
        printMSE(stoch_mse, "Linear-regression-MSE_with_Z_Normalization--Stochastic_Gradient");
        printMSE(batch_mse, "Linear-regression-MSE_with_Z_Normalization--zBatch_Gradient");
        System.out.println("Please Check the output folder");
    }
    
    

    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    private static Matrix getBeta(Matrix trainX, Matrix trainY) {

        //new_trainX.print(5, 6);
        // beta = (X^T*X)^ -1)*(X^T*Y)
        
        Matrix beta =  ( ( ( trainX.transpose() ).times(trainX) ).inverse() ).times( (trainX.transpose()).times(trainY)  );
        
        /*beta.print(5,5);
        System.out.println( trainX.getRowDimension() + "   " + trainX.getColumnDimension() );
        System.out.println( new_trainX.getRowDimension() +"   " +  new_trainX.getColumnDimension() );

        System.out.println( new_trainX.getRowDimension() +"   " +  new_trainX.getColumnDimension() );
        
        System.out.println( beta.getRowDimension() + "   "+beta.getColumnDimension());
        */
        
        return beta;
    }
    
    private static Matrix getBetaforStochastic_Gradient(Matrix train_x, Matrix train_y) {
        
        int length = train_x.getColumnDimension();
        int width = train_x.getRowDimension();
        
        //Matrix beta = Matrix(1, length, 1.0d);
        Matrix beta = new Matrix(1, length,1);
        
        for(int counter=0; counter<width; counter++){
            double error = train_y.get(counter ,0) -
                            train_x.getMatrix(counter,counter,0, length- 1)
                              .times(beta.transpose()).get(0, 0);

            beta = beta.plus(train_x.getMatrix(counter,counter,0,length-1)
                                .times(2* 0.001*error) );
            }
            return beta.transpose();
	}
        
    private static Matrix getBetaforBatch_Gradient(Matrix train_x, Matrix train_y) {
        
        int length = train_x.getColumnDimension();
        int width = train_x.getRowDimension();
			
        //Matrix beta = Matrix(1, length, 1.0d);
        Matrix beta = new Matrix(1, length,1);
        Matrix beta1 = new Matrix(1, length,1);
        
        for (int counter = 0; ; counter++) {

            double error = train_x.getMatrix(counter,counter,0, length- 1)
                              .times(beta.transpose()).get(0, 0)
                                                    - train_y.get(counter ,0);

            beta = beta.minus(train_x.getMatrix(counter,counter,0,length-1)
                                                        .times(2* 0.01*error) );
                        
            if(counter!=0 && Math.abs(
                                 getMatrixMean(beta.transpose())[0]
                             -   getMatrixMean(beta1.transpose())[0] 
                                )<= 0.00001   )
                    break;
                beta1 = beta;
            
            if(counter == width-2)
                counter=0;
        }
        
        return beta.transpose();
    }
    
    private static Matrix addMatrixColumnWith1(Matrix matrix) {
        
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension() + 1;
        
        Matrix new_matrix = new Matrix(rows, cols);
        
        for (int row = 0; row < rows; row++){
            for (int col = 0; col < cols; col++){
                if (col == 0)
                    new_matrix.set(row, col, 1);
                else 
                    new_matrix.set(row, col, matrix.get(row, col-1));   
                }
        }
        
        return new_matrix;
    }
    
    private static double getMeanSquareError(Matrix predictedY, Matrix test_y) {
        
        Matrix var = predictedY.minus(test_y);
        
        Matrix mse = var.transpose().times(var);
        
        return ( mse.get(0,0)/ predictedY.getRowDimension() );
    }
    
    private static double[] getMatrixMean(Matrix matrix) {
        
        int length = matrix.getColumnDimension();
        int width = matrix.getRowDimension();
        
        double[] mean = new double[length];
        
        for(int row=0; row<length; row++){
            double columnsum = 0d;
            for (int col = 0; col<width; col++)
                columnsum += matrix.get(col, row);
            
            mean[row] = columnsum/width;
        }
        
        return mean;
    }
    //z(i)=(x(i)−μ(i))/σ(i)
 
    // Here x(i) is the value of i'th feature for input vector x.
    // μ(i) is mean of i'th feature values in training data.
    // σ(i) is the standard deviation of i'th feature values in training data.
    private static double[] getMatrixSD(Matrix matrix){
        
        double[] mean = getMatrixMean(matrix); 
        int length = matrix.getColumnDimension();
        int width = matrix.getRowDimension();
        Matrix var_matrix = new Matrix(width, length);
        
        for(int row =0; row< length; row++)
            for(int col=0; col<width;col++){
                double variance = matrix.get(col, row) -mean[row];
                var_matrix.set(col,row,(variance*variance));
            }
        
        double[] stan_devi = getMatrixMean(var_matrix);
        
        for(int counter =0; counter< stan_devi.length;counter++)
            stan_devi[counter] = Math.sqrt(stan_devi[counter]);
        
        return stan_devi;

    }
    
    public static Matrix zNormalize(Matrix matrix) {
        
        double[] matrixMean =getMatrixMean(matrix);
        double[] matrixSD = getMatrixSD(matrix);
        int length = matrix.getColumnDimension();
        int width = matrix.getRowDimension();
        
        for (int row =0; row <length; row++){
            for(int col = 0; col < matrix.getRowDimension(); col++) {
                if(matrixSD[row] > 0)
                    matrix.set(col,row,(matrix.get(col,row) -matrixMean[row])/matrixSD[row]);
                else
                    matrix.set(col,row,(matrix.get(col,row) -matrixMean[row]));
            }
        }
        return matrix;
    }

    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput(Matrix matrix, String fileName) throws IOException {
        //FileWriter fStream = new FileWriter("output/linear_regression/linear-regression-output.txt");     // Output File
        FileWriter fStream = new FileWriter("output/linear_regression/"+ fileName +".txt");
        BufferedWriter out = new BufferedWriter(fStream);
        
        for (int row =0; row<matrix.getRowDimension(); row++) {
            out.write(String.valueOf(matrix.get(row, 0)));
            out.newLine();
        }
        out.close();
        
    }
    
    public static void printMSE(double mse, String fileName) throws IOException {
        //FileWriter fStream = new FileWriter("output/linear_regression/linear-regression-output.txt");     // Output File
        FileWriter fStream = new FileWriter("output/linear_regression/"+ fileName +".txt");
        BufferedWriter out = new BufferedWriter(fStream);
        out.write(mse+"");
        out.close();
        
    }
}
