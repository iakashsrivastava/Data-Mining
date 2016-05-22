import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import Jama.Matrix;
import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.util.ArrayList;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 6/22/12
 * Time: 11:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Logistic {

    /** the learning rate */
    private double rate;

    /** the weight to learn */
    private double[] weights;
    
    private double bias; 

    /** the number of iterations */
    private int ITERATIONS = 6000;

    private static final int numberOfFolds = 5;
    
    private double EPSI = Double.longBitsToDouble(971l << 52);

    public Logistic(int n) {
        this.rate = 0.001;
        weights = new double[n];
        bias = 0;
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void train(List<Instance> instances) {
        double lik = 0.0;
        double prev_lik = 0.0;
        int iteration =0;
        do{
            prev_lik =lik;
            lik =0.0;

            double []predictVec = new double[instances.size()];
            for (int i=0; i<instances.size(); i++) {
                double[] x = instances.get(i).getX();
                double predicted = classify(x);
                predictVec[i] = predicted;
            }
	//update weights and bias
        /****************Please Fill Missing Lines Here*****************/

        int col = instances.get(0).dimension;
        int noOfInstances = instances.size();

        Matrix firstDerivative = new Matrix(1, col +1);
        Matrix secondDerivative = new Matrix(col +1, col +1);

        for(int counter=0; counter<noOfInstances; counter++){

            double[] x = instances.get(counter).getX();
            double[] xPlus = new double[x.length + 1];

            xPlus[0]= 1;
            System.arraycopy(x, 0, xPlus, 1, x.length);

            Matrix firstDerivativeTemp = new Matrix(xPlus, 1);
            firstDerivative = firstDerivative.plus(firstDerivativeTemp.times
                                        (instances.get(counter).label - predictVec[counter]));

            Matrix matX = new Matrix(xPlus, 1);
            Matrix matXTran = matX.transpose();

            Matrix newX = matXTran.times(matX);
            newX = newX.times(predictVec[counter]);
            newX = newX.times(1 - predictVec[counter]);

            secondDerivative = secondDerivative.minus(newX);
        }

        Matrix secondDerivativeTemp = secondDerivative.minus(Matrix.identity(col+1, col+1));

        Matrix res = firstDerivative.times(secondDerivativeTemp.inverse());
        double[] res1D = res.getRowPackedCopy();
        bias -= res1D[0];

        for(int counter = 1; counter< res1D.length ; counter++)
            weights[counter-1] -= res1D[counter];


	//calculate log likelihood function 
	/****************Please Fill Missing Lines Here*****************/

        for(int counter =0; counter<noOfInstances;counter++){
            double[] x = instances.get(counter).getX();
            lik += ((getLogit(x) * instances.get(counter).label) - Math
                    .log(1 + Math.exp(getLogit(x))));
        }

        System.out.println("iteration: "  + ++iteration+ " " + Arrays.toString(weights) + " mle: " + lik);
        }while( (Math.abs(lik - prev_lik ) > EPSI) && (iteration < ITERATIONS)); // whichever is less
    }

    private double getLogit(double[] x) {
        double logit = bias;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x[i];
        }
        return logit;
    }

    private double classify(double[] x) {
        double logit = bias;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }


    public static void main(String... args) throws FileNotFoundException {

        double averageAccuracy =0;
        String str ="";

        List<Instance> instances = DataSet.readDataSet("data/data.txt");
        List<Instance> trainInstance, testInstance;

        Collections.shuffle(instances);

        int noOfInstances = instances.size();
        int avgSize = noOfInstances/numberOfFolds;


        for(int counter =0; counter<numberOfFolds; counter++){
            trainInstance = new ArrayList<Instance>();
            testInstance = new ArrayList<Instance>();

            int start = counter*avgSize;
            int end = start+avgSize-1;

            for(int innCounter =0; innCounter<noOfInstances; innCounter++)
                if(innCounter>= start && innCounter< end)
                    testInstance.add(instances.get(innCounter));
                else
                    trainInstance.add(instances.get(innCounter));

            Logistic logistic = new Logistic(instances.get(0).getDimension());
            logistic.train(instances);

            int correctInstances =0;

            for(Instance instance: testInstance) {
                int instantceval = (logistic.classify(instance.x) > .5) ? 1 : 0;
                if (instance.label == instantceval)
                    ++correctInstances;
            }
            double accuracy = (double) correctInstances / testInstance.size();
            str +="Accuracy After fold " + (counter+1) +" : " + accuracy+"\n";
            averageAccuracy += accuracy;

        }

        System.out.println(str);
        System.out.println("Average accuracy after five folds: "+ averageAccuracy/(double)numberOfFolds);
        //toy test
        //double[] testPoint = {63.0278175,22.55258597,39.60911701,40.47523153,98.67291675,-0.254399986};
        //System.out.println("prob(1|testPoint) = " + logistic.classify(testPoint));

    }
}
