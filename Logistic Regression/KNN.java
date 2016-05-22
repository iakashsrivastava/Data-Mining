/**
 * Created by akash on 2/15/16.
 */
import java.util.Comparator;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import Jama.Matrix;
import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.util.ArrayList;

public class KNN {

    private static final int numberOfFolds= 5;
    private static final int k= 7;

    public static double getClassifierValue(int k, List<Instance> trainInstances, Instance testInstance){

        int zeros = 0;
        int ones = 0;
        int counter = 0;

        InstanceNode[] trainInstancesList = new InstanceNode[ trainInstances.size()];

        for (Instance instance : trainInstances) {

            InstanceNode instanceNode = new InstanceNode();
            instanceNode.setLabel(instance.label);
            instanceNode.setDistance( euclideanDistance(instance, testInstance) );

            trainInstancesList[counter] = instanceNode;
            counter++;
        }

        //
        Arrays.sort(trainInstancesList, new InstanceNodeComparator());

        for(counter =0; counter<k; counter++)
            if (trainInstancesList[counter].getLabel() == 1)
                ones++;
            else
                zeros++;

        int result = (zeros > ones) ? 0 : 1;
        return result;

    }



    public static double euclideanDistance(Instance trainInstance, Instance testInstance){

        double distance = 0.0d;
        for(int counter =0; counter<trainInstance.dimension; ++counter)
            distance += Math.pow((trainInstance.x[counter] - testInstance.x[counter]), 2.0);

        return Math.sqrt(distance);
    }


    public static void main(String... args) throws FileNotFoundException {

        double averageAccuracy =0;

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


            int correctInstances =0;

            for(Instance instance: testInstance) {
                int instantceval = ( getClassifierValue(k, trainInstance, instance) > .5) ? 1 : 0;
                if (instance.label == instantceval)
                    ++correctInstances;
            }
            double accuracy = (double) correctInstances / testInstance.size();
            System.out.println("Accuracy After fold " + (counter+1) +" : " + accuracy);
            averageAccuracy += accuracy;

        }

        System.out.println("Average accuracy after five folds: "+ averageAccuracy/(double)numberOfFolds);
        //toy test
        //double[] testPoint = {63.0278175,22.55258597,39.60911701,40.47523153,98.67291675,-0.254399986};
        //System.out.println("prob(1|testPoint) = " + logistic.classify(testPoint));

    }


}
