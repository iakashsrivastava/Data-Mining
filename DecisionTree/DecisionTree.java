package DecisionTree;

import java.io.*;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import Utility.Utility;

/*

Class for constructing an unpruned decision tree based on
the ID3 algorithm. Can only deal with nominal attributes.
No missing values allowed. Empty leaves may result in unclassified instances.

*/

public class DecisionTree  {
    //The node's successors.
    private DecisionTree[] m_Successors;
    //Attribute used for splitting.
    private Attribute m_Attribute;
    //Class value if node is leaf.
    private double m_ClassValue;
    //Class distribution if node is leaf.
    private double[] m_Distribution;
    // Class attribute of data set.
    private Attribute m_ClassAttribute;
    // Number of Folds
    private final int FOLDS = 5;
    // 
    private double averageAccuracy;

    public DecisionTree() {
    }
    //Builds decision tree classifier.
    public void buildClassifier(Instances data, boolean isGainRatio) throws Exception {
        data = new Instances(data);
        this.makeTree(data,isGainRatio);
    }

    private void makeTree(Instances data, boolean isGainRatio) throws Exception {
        if(data.numInstances() == 0) {
            this.m_Attribute = null;
            this.m_ClassValue = Instance.missingValue();
            this.m_Distribution = new double[data.numClasses()];
        } else {
            double[] infoGains = new double[data.numAttributes()];

            Attribute splitData;
            for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[splitData.index()] = this.computeInfoGain(data, splitData,isGainRatio)) {
                splitData = (Attribute)attEnum.nextElement();
            }

            this.m_Attribute = data.attribute(Utils.maxIndex(infoGains));
            if(Utils.eq(infoGains[this.m_Attribute.index()], 0.0D)) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance j;
                for(Enumeration var6 = data.enumerateInstances(); var6.hasMoreElements(); ++this.m_Distribution[(int)j.classValue()]) {
                    j = (Instance)var6.nextElement();
                }

                Utils.normalize(this.m_Distribution);
                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();
            } else {
                Instances[] var7 = this.splitData(data, this.m_Attribute);
                this.m_Successors = new DecisionTree[this.m_Attribute.numValues()];

                for(int var8 = 0; var8 < this.m_Attribute.numValues(); ++var8) {
                    this.m_Successors[var8] = new DecisionTree();
                    this.m_Successors[var8].makeTree(var7[var8],isGainRatio);
                }
            }
        }
    }
    //Classifies a given test instance using the decision tree.
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if(instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DecisionTree: no missing values, please.");
        } else {
            return this.m_Attribute == null?this.m_ClassValue:this.m_Successors[(int)instance.value(this.m_Attribute)].classifyInstance(instance);
        }
    }

    public String toString() {
        return this.m_Distribution == null && this.m_Successors == null?"DecisionTree: No model built yet.":"DecisionTree\n\n" + this.toString(0);
    }
    //Computes information gain for an attribute.
    private double computeInfoGain(Instances data, Attribute att, 
                                        boolean isGainRatio) throws Exception {
        double infoGain = this.computeEntropy(data);
        Instances[] splitData = this.splitData(data, att);

        /****************Please Fill Missing Lines Here*****************/
        
        double totalInstances = 0d;
        double infoGain_Instance = 0d;
        double gain = 0d;
        
        for (Instances ins : splitData)
            totalInstances += ins.numInstances();
        
        for (Instances ins : splitData) {
            double getEntropy = this.computeEntropy(ins);
            
            infoGain_Instance += ( (double)ins.numInstances() /
                                                   totalInstances) * getEntropy;
        }
        
        gain = infoGain - infoGain_Instance;
        //System.out.println(gain);
        if (isGainRatio && infoGain_Instance ==0) 
            return 0;
        else if(isGainRatio)
            return gain / infoGain_Instance;
        else
            return gain;
    }
    //Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];

        Instance entropy;
        for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++classCounts[(int)entropy.classValue()]) {
            entropy = (Instance)instEnum.nextElement();
        }

        double totalEntropy = 0.0D;
        int classNum = data.numClasses();
        double [] classProbVec = new double[classNum];
        
        for(int j = 0; j < classNum; ++j) {
            if(classCounts[j] > 0.0D) {
                classProbVec[j]= classCounts[j]/data.numInstances();
            }
            else
            	classProbVec[j]=0;
        }

        /****************Please Fill Missing Lines Here*****************/
        
        for(int counter = 0; counter < classNum; counter++){
            if(classCounts[counter] > 0.0D) {
                totalEntropy += (-1) * classProbVec[counter] *
                                        ( Math.log(classProbVec[counter]) 
                                                            / Math.log(2) );
                                    
            }
        }
        
        return totalEntropy;

    }
    //Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];

        for(int instEnum = 0; instEnum < att.numValues(); ++instEnum) {
            splitData[instEnum] = new Instances(data, data.numInstances());
        }

        Enumeration var6 = data.enumerateInstances();

        while(var6.hasMoreElements()) {
            Instance i = (Instance)var6.nextElement();
            splitData[(int)i.value(att)].add(i);
        }

        for(int var7 = 0; var7 < splitData.length; ++var7) {
            splitData[var7].compactify();
        }
        return splitData;
    }

    private String toString(int level) {
        StringBuffer text = new StringBuffer();
        if(this.m_Attribute == null) {
            if(Instance.isMissingValue(this.m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + this.m_ClassAttribute.value((int)this.m_ClassValue));
            }
        } else {
            for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                text.append("\n");

                for(int i = 0; i < level; ++i) {
                    text.append("|  ");
                }

                text.append(this.m_Attribute.name() + " = " + this.m_Attribute.value(j));
                text.append(this.m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    public void decisionTree() throws Exception {
        
        BufferedReader file = Utility.readFile("data/decision_tree/weather-nominal.arff");
        Instances data = new Instances(file);
        int cIdx=data.numAttributes()-1;
        data.setClassIndex(cIdx);
        buildClassifier(data,false);
        printOutput(data);
        
    }
    
    public void decisionTree(boolean isGainRatio) throws Exception {
        
        averageAccuracy = 0.0D;
        BufferedReader file = 
                Utility.readFile("data/decision_tree/Congressional_Voting_Records.arff");
        
        Instances data = new Instances(file);
        int cIdx=data.numAttributes()-1;
        data.setClassIndex(cIdx);
       
        for (int counter=0; counter<FOLDS; counter++) {
            
            Instances trainData = data.trainCV(5,counter);
            Instances testData = data.testCV(5,counter);
            
            buildClassifier(trainData, isGainRatio);
            double accuracy = printOutput(testData, counter, isGainRatio);
            averageAccuracy += accuracy;
            
        }
        
        if (isGainRatio)
            System.out.println("Average accuracy with Gain Ratio " 
                                                        +averageAccuracy/FOLDS);
        else
            System.out.println("Average accuracy with Info Gain "
                                                        +averageAccuracy/FOLDS);
	}

    private void printOutput(Instances data) throws IOException, NoSupportForMissingValuesException {
        FileWriter fStream = new FileWriter("output/decision_tree/decision-tree-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);

        for(int index =0; index<data.numInstances();index++) {
            Instance testRowInstance = data.instance(index);
            double prediction =classifyInstance(testRowInstance);
            out.write(data.classAttribute().value((int) prediction));
            out.newLine();
        }
        out.close();
    }
    
    private double printOutput(Instances data, int counter, boolean isGainRatio) 
            throws IOException, NoSupportForMissingValuesException {
        
        double predictions = 0d;
        String fileName;
        
        if(isGainRatio)
            fileName = "output/decision_tree/decision-tree-output_withGainRatio_after_fold-"
                    +counter+".txt";
        else
            fileName = "output/decision_tree/decision-tree-output_withoutGainRatio_after_fold-"
                    +counter+".txt";
        
        FileWriter fStream = new FileWriter(fileName);     // Output File
        BufferedWriter out = new BufferedWriter(fStream);

        for(int index =0; index<data.numInstances();index++) {
            Instance testRowInstance = data.instance(index);
            double prediction =classifyInstance(testRowInstance);
            
            if (testRowInstance.classValue() == prediction)
		predictions++;
		
            out.write(data.classAttribute().value((int) prediction));
            out.newLine();
        }
        out.close();
        return predictions / (double) data.numInstances();
    }    
}
