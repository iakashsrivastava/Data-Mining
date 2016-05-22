package edu.neu.dm.hw3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class DBSCAN {
	static double e = 0.0;
	static int minPts = 3;
	static int noOfLabels=0;
	static int dataset=0;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int seed = 71;
		System.out.println("For dataset1");
		List<DataPoints> dataSet = KMeans.readDataSet("data/dataset1.txt");
		Collections.shuffle(dataSet, new Random(seed));
		noOfLabels = DataPoints.getNoOFLabels(dataSet);
		e = getEpsilon(dataSet);
		System.out.println("Esp :"+e);
		dbscan(dataSet);
		
		System.out.println("\nFor dataset2");
		dataSet = KMeans.readDataSet("data/dataset2.txt");
		Collections.shuffle(dataSet, new Random(seed));
		noOfLabels = DataPoints.getNoOFLabels(dataSet);
		e = getEpsilon(dataSet);
		System.out.println("Esp :"+e);
		dbscan(dataSet);
		
		System.out.println("\nFor dataset3");
		dataSet = KMeans.readDataSet("data/dataset3.txt");
		Collections.shuffle(dataSet, new Random(seed));
		noOfLabels = DataPoints.getNoOFLabels(dataSet);
		e = getEpsilon(dataSet);
		System.out.println("Esp :"+e);
		dbscan(dataSet);
	}

	private static double getEpsilon(List<DataPoints> dataSet) {
		List<Double> distances = new ArrayList<Double>();
		double sumOfDist = 0d;
		for(int i=0; i < dataSet.size(); i++) {
			DataPoints point = dataSet.get(i);
			double dist = 0d;
			for(int j=0; j < dataSet.size(); j++) {
				if(i == j)
					continue;
				DataPoints pt = dataSet.get(j);
				dist = getEuclideanDist(point.x, point.y, pt.x, pt.y);
				distances.add(dist);
			}
			Collections.sort(distances);
			sumOfDist += distances.get(7);
			distances.clear();
		}
		return sumOfDist/dataSet.size();
	}
	
	private static void dbscan(List<DataPoints> dataSet) {
		List<Set<DataPoints>> clusters = new ArrayList<Set<DataPoints>>();
		Set<DataPoints> visited = new HashSet<DataPoints>();
		Set<DataPoints> noise = new HashSet<DataPoints>();
		
		//Iterate over data points
		for(int i=0; i < dataSet.size(); i++) {
			DataPoints point = dataSet.get(i);
			if(visited.contains(point))
				continue;
			visited.add(point);
			List<DataPoints>  N = new ArrayList<DataPoints>();
			int minPtsNeighbours = 0;
			
			//check which point satisfies minPts condition 
			for(int j=0; j < dataSet.size(); j++) {
				if(i == j)
					continue;
				DataPoints pt = dataSet.get(j);
				double dist = getEuclideanDist(point.x, point.y, pt.x, pt.y);
				if(dist <= e) {
					minPtsNeighbours++;
					N.add(pt);
				}
			}
			
			if(minPtsNeighbours >= minPts) {
				Set<DataPoints> cluster = new HashSet<DataPoints>();
				cluster.add(point);
				point.isAssignedToCluster = true;
				
				for(int j=0; j < N.size(); j++) {
					DataPoints point1 = N.get(j);
					int minPtsNeighbours1 = 0;
					List<DataPoints> N1 = new ArrayList<DataPoints>();
					if(!visited.contains(point1)) {
						visited.add(point1);
						for(int l=0; l < dataSet.size(); l++) {
							DataPoints pt = dataSet.get(l);
							double dist = getEuclideanDist(point1.x, point1.y, pt.x, pt.y);
							if(dist <= e) {
								minPtsNeighbours1++;
								N1.add(pt);
							}
						}
						if(minPtsNeighbours1 >= minPts) {
							removeDuplicates(N, N1);
							
						} else {
							N1.removeAll(N1);
						}
					}
					//Add point1 is not yet member of any other cluster then add it to cluster
					if(!point1.isAssignedToCluster) {
						cluster.add(point1);
						point1.isAssignedToCluster = true;
					}
				}
				//add cluster to the list of clusters
				clusters.add(cluster);
			} else {
				noise.add(point);
			}
			
			N.removeAll(N);
		}
		//List clusters
		System.out.println("Number of clusters formed :"+clusters.size());
		System.out.println("Noise points :"+noise.size());
		
		//Calculate purity
		int[] maxLabelCluster = new int[clusters.size()];
		for (int j = 0; j < clusters.size(); j++) {
			maxLabelCluster[j] = KMeans.getMaxClusterLabel(clusters.get(j));
		}
		double purity = 0d;
		for(int j=0; j< clusters.size(); j++) {
			purity += maxLabelCluster[j];
		}
		purity = purity/dataSet.size();
		System.out.println("Purity is :"+purity);
		
		double[][] nmiMatrix = DataPoints.getNMIMatrix(clusters, noOfLabels);
		double nmi = DataPoints.calcNMI(nmiMatrix);
		System.out.println("NMI :"+nmi);
		
		DataPoints.writeToFile(noise, clusters, "output/DBSCAN_dataset"+ ++dataset +".csv");


	}

	private static void removeDuplicates(List<DataPoints> n, List<DataPoints> n1) {
		// TODO Auto-generated method stub
		for(DataPoints point : n1) {
			boolean isDup = false;
			for(DataPoints point1 : n) {
				if(point1.equals(point))
					isDup = true;
			}
			if(!isDup)
				n.add(point);
		}
		
	}

	private static double getEuclideanDist(double x1, double y1, double x2, double y2) {
		double dist = Math.sqrt(Math.pow((x2-x1), 2) + Math.pow((y2-y1), 2));
		return dist;
	}

}
