import java.util.Comparator;

/**
 * Created by akash on 2/15/16.
 */
public class InstanceNodeComparator implements Comparator<InstanceNode> {
    @Override
    public int compare(InstanceNode instance1, InstanceNode instance2) {

        if ((instance1 == null) || (instance1 == null))
            return 0;
        else if (instance1.getDistance()> instance2.getDistance() )
            return 1;
        else if (instance1.getDistance()< instance2.getDistance() )
            return -1;
        else
            return 0;
    }

}
