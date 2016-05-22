package edu.neu.dm.hw3;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import edu.neu.dm.hw3.*;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;


public class GMM {
	public static List<DataPoints> dataSet;
	public static int K=0;
	public static double[][] mean = new double[3][2];
	public static double[][] stdDev = new double[3][2];
	public static double[][][] coVariance = new double[3][2][2];
	public static double[][] W;
	public static double[] w;
	static int dataset=0;
	
	public static void main(String[] args)throws Exception {
		// TODO Auto-generated method stub
		dataSet = DataPoints.readDataSet("data/dataset1.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		//Collections.shuffle(dataSet);
		W = new double[dataSet.size()][K];
		w = new double[K];
		GMM();

		dataSet = DataPoints.readDataSet("data/dataset2.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		//Collections.shuffle(dataSet);
		W = new double[dataSet.size()][K];
		w = new double[K];
		GMM();

		dataSet = DataPoints.readDataSet("data/dataset3.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		//Collections.shuffle(dataSet);
		W = new double[dataSet.size()][K];
		w = new double[K];
		GMM();
	}
	
	public static void GMM()throws Exception {
		List<Set<DataPoints>> clusters = new ArrayList<Set<DataPoints>>();
		int k =0;
		while(k < K) {
			Set<DataPoints> cluster = new HashSet<DataPoints>();
			clusters.add(cluster);
			k++;
		}
		
		//Initially randomly assign points to clusters
		int i = 0;
		for(DataPoints point : dataSet) {
			clusters.get(i%K).add(point);
			i++;
		}
		
		for(int m=0; m< K; m++) {
			w[m] = 1.0/K;
		}
		//Get Intial mean
		//DataPoints.getMean(clusters, mean);
		//DataPoints.getStdDeviation(clusters, mean, stdDev);
		//DataPoints.getCovariance(clusters, mean, stdDev, coVariance);

        getMean(clusters);
        getStdDeviation(clusters);
        getCovariance(clusters);

		int len=0;
		double mle_old =0d, mle_new = 0d;
		while(true) {
			mle_old = likelyHood();
			Estep();
			Mstep(clusters);
			len++;
			mle_new = likelyHood();
			
			//convergence condition
            /*
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            System.out.println(" =====================================================================================");
            */
            System.out.println(Math.abs(mle_new - mle_old)/Math.abs(mle_old));
			if((Math.abs(mle_new - mle_old)/Math.abs(mle_old)) < 0.0001)
				break;
		}
		System.out.println("Number of Iterations = "+len);
		System.out.println("\nAfter Calculations");
		System.out.println("Final mean = ");
		printArray(mean);
		System.out.println("\nFinal covariance = ");
		print3D(coVariance);

		//Calculate purity
		int[] maxLabelCluster = new int[clusters.size()];
		for (int j = 0; j < clusters.size(); j++) {
			maxLabelCluster[j] = getMaxClusterLabel(clusters.get(j));
		}
		double purity = 0d;
		for(int j=0; j< clusters.size(); j++) {
			purity += maxLabelCluster[j];
		}
		purity = purity/dataSet.size();
		System.out.println("Purity is :"+purity);

		double[][] nmiMatrix = DataPoints.getNMIMatrix(clusters, K);
		double nmi = DataPoints.calcNMI(nmiMatrix);
		System.out.println("NMI :"+nmi);


		//write clusters to file for plotting
			String filename = "GMM_dataset" + ++dataset + ".csv";
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File("output/" + filename)));
			for (int w = 0; w < K; w++) {
				System.out.println("Cluster " + w + " size :" + clusters.get(w).size());
				//bw.write("\nFor Cluster "+ (w+1) + "\n");
				for (DataPoints point : clusters.get(w))
					bw.write(point.x + "," + point.y + "\n");

				bw.write("\n\n");
			}

		bw.close();
	}

	public static int getMaxClusterLabel(Set<DataPoints> cluster) {
		Map<Integer, Integer> labelCounts = new HashMap<Integer, Integer>();
		for(DataPoints point : cluster) {
			if(!labelCounts.containsKey(point.label)) {
				labelCounts.put(point.label, 1);
			} else {
				labelCounts.put(point.label, labelCounts.get(point.label) + 1);
			}
		}
		int max = Integer.MIN_VALUE;
		for(int label : labelCounts.keySet()) {
			if(max < labelCounts.get(label))
				max = labelCounts.get(label);
		}
		return max;
	}

    public static void getMean(List<Set<DataPoints>> clusters){
        double mx=0.0d;
        double my=0.0d;
        int i=0;
        for(Set<DataPoints> cluster : clusters){
            int size =cluster.size();
            if(size == 0){
                mean[i][0] = 0.0d;
                mean[i][1] = 0.0d;
            }
            for(DataPoints point : cluster) {
                mx += point.x;
                my += point.y;
            }
            mx = mx/size;
            my = my/size;
            mean[i][0] = mx;
            mean[i][1] = my;
            i++;
        }

    }

    private static void getStdDeviation(List<Set<DataPoints>> clusters){

        int i=0;
        for(Set<DataPoints> cluster : clusters){
            double mx= mean[i][0];
            double my= mean[i][1];
            double sdx=0.0d;
            double sdy=0.0d;
            int size =cluster.size();
            if(size == 0){
                stdDev[i][0] = 0.0d;
                stdDev[i][1] = 0.0d;
            }

            for(DataPoints point : cluster) {
                sdx += Math.pow( Math.abs(mx-point.x),2);
                sdy += Math.pow( Math.abs(my-point.y),2);
            }
            sdx = Math.sqrt(sdx/size);
            sdy = Math.sqrt(sdy/size);
            stdDev[i][0] = sdx;
            stdDev[i][1] = sdy;
            i++;
        }
    }

    private static void getCovariance(List<Set<DataPoints>> clusters){
        int j=0;
        for(Set<DataPoints> cluster : clusters){
            coVariance[j][0][0] = stdDev[j][0];
            coVariance[j][1][1] = stdDev[j][1];
            coVariance[j][0][1] = coVariance[j][1][0] = 1/cluster.size();
            j++;
        }
    }

	public static void Estep() {
		for(int i=0; i < dataSet.size(); i++) {
			double denominator = 0d;
			
			for(int j=0; j < K; j++) {
				MultivariateNormalDistribution gaussian;
				gaussian = new MultivariateNormalDistribution(mean[j], coVariance[j]);
				double numerator = w[j] * gaussian.density(new double[]{dataSet.get(i).x, dataSet.get(i).y});
                denominator += numerator;
				W[i][j] = numerator;
			}
			
			//normalize W[i][j] into probabilities

/****************Please Fill Missing Lines Here*****************/
            for(int l=0; l < dataSet.size(); l++) {
                for(int m=0; m < K; m++) {
                    MultivariateNormalDistribution gaussian;
                    gaussian = new MultivariateNormalDistribution(mean[m], coVariance[m]);
                    double numerator = w[m] * gaussian.density(new double[]{dataSet.get(l).x, dataSet.get(l).y});

                    W[l][m] = numerator/denominator;
                }
            }

		}          
	}
	
	public static void Mstep(List<Set<DataPoints>> clusters) {
		//get 
		for(int j=0; j < K; j++) {
			double denominator = 0d;
			double numerator = 0d;
			double numerator1 = 0d;
			double cov_xy = 0d;
			double updatedMean1 = 0d, updatedMean2 = 0d;


			for(int i=0; i < dataSet.size(); i++) {
				denominator += W[i][j];
				numerator += W[i][j] * Math.pow((dataSet.get(i).x - mean[j][0]), 2);
				numerator1 += W[i][j] * Math.pow((dataSet.get(i).y - mean[j][1]), 2);
				//cov_xy +=?

/****************Please Fill Missing Lines Here*****************/
                cov_xy += W[i][j] * (dataSet.get(i).x - mean[j][0]) * (dataSet.get(i).y - mean[j][1]) ;
				
				updatedMean1 += W[i][j] * dataSet.get(i).x;
				updatedMean2 += W[i][j] * dataSet.get(i).y;
			}
			stdDev[j][0] = numerator / denominator;
			stdDev[j][1] = numerator1 / denominator;
			//update w[j]

/****************Please Fill Missing Lines Here*****************/

            for(int l=0; l < dataSet.size(); l++) {
                for(int m=0; m < K; m++) {
                    w[m] +=W[l][m];
                }
            }

            for(int m=0; m < K; m++) {
                w[m] /=dataSet.size();
            }



			//update mean
			mean[j][0] = updatedMean1 / denominator;
			mean[j][1] = updatedMean2 / denominator;

            getStdDeviation(clusters);

			//update covariance matrix
			coVariance[j][0][0] = stdDev[j][0];
			coVariance[j][1][1] = stdDev[j][1];
			coVariance[j][0][1] = coVariance[j][1][0] = cov_xy/denominator;
		}	
	}
	
	public static double likelyHood() {
		double liklyhood = 0d;
		for(int i=0; i < dataSet.size(); i++) {
			double numerator = 0d;
			for(int j=0; j < K; j++) {
				MultivariateNormalDistribution gaussian;
/*
                System.out.println(mean[j].length+"       "+coVariance[j].length);
                for(int o=0;o<mean[j].length;o++)
                    System.out.println("mean ["+j+"]["+o+"]   "+mean[j][o]);

                for(int o=0;o<stdDev[j].length;o++)
                    System.out.println("standard ["+j+"]["+o+"]   "+stdDev[j][o]);

                for(int o=0;o<coVariance[j].length;o++)
                    for(int h=0;h<coVariance[j][o].length;h++) {
                        System.out.println("covariance [" + o + "][" + h + "]   " + coVariance[j][o][h]);
                    }
*/
                gaussian = new MultivariateNormalDistribution(mean[j], coVariance[j]);
				numerator += w[j] * gaussian.density(new double[]{dataSet.get(i).x, dataSet.get(i).y});
			}
            //System.out.println(numerator);
            liklyhood += Math.log(numerator);
		}
       // System.out.println("return     "+ liklyhood);
		return liklyhood;
	}
	
	public static void printArray(double mat[][]) {
		for(int i=0; i < mat.length; i++) {
			for(int j=0; j < mat[i].length; j++) {
				System.out.print(mat[i][j] +  "  ");
			}
			System.out.println();
		}
	}
	
	public static void print3D(double mat[][][]) {
		for(int i=0; i < mat.length; i++) {
			System.out.println("For Cluster : "+(i+1));
			for(int j=0; j < mat[i].length; j++) {
				for(int k=0; k < mat[i][j].length; k++) {
					System.out.print(mat[i][j][k] +  "  ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}
	
	// Helper function to plot points in xcel
	public void plot() throws IOException {
		FileWriter fStream = new FileWriter("xcel.csv");
		BufferedWriter out = new BufferedWriter(fStream);
		for (int i = 0; i < dataSet.size(); i++) {
			DataPoints point = dataSet.get(i);
			int label = 0;
			label = point.label;
			out.write(String.valueOf(point.x) + ","
					+ String.valueOf(point.y) + ","
					+ String.valueOf(label));
			out.newLine();
		}
		out.close();
	}

}
