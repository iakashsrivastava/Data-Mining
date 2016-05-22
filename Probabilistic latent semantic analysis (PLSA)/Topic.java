/**
 * Created by akash on 3/4/16.
 */
import java.util.Comparator;

public class Topic implements Comparator<Topic> ,Comparable<Topic>{
    int topic;
    double prob;

    public Topic() {

    }

    public Topic(int topic, double prob) {
        this.topic = topic;
        this.prob = prob;
    }

    @Override
    public int compareTo(Topic arg0) {
        //System.out.println(arg0);
        return Double.compare(arg0.prob, prob);
    }

    @Override
    public int compare(Topic arg0, Topic arg1) {
        return Double.compare(arg1.prob, arg0.prob);
    }
}
