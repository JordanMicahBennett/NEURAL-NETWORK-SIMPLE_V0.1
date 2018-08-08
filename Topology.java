//author: Jordan Micah Bennett
import java.util.ArrayList;

public class Topology extends ArrayList <Integer>
{
    private String description; //specification of neural network layer architecture

    public Topology ( String description )
    {
        this.description = description;
        generateTopology ( );
    }

    public void generateTopology ( )
    {
        String [ ] segmentedValues = description.split ( "," );
        
        for ( int sVI = 0; sVI < segmentedValues.length; sVI ++) //sVI = segment values iterator
            add ( Integer.parseInt ( segmentedValues [sVI] ) );
    }
}