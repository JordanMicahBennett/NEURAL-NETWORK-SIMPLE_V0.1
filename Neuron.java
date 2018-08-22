//author: Jordan Micah Bennett
import java.util.Random;
import java.util.ArrayList;

public class Neuron
{
    private double eta;
    private double outcome;
    private double alpha;
    private double gradient;
    private ArrayList<Synapse> synapses = new ArrayList <Synapse> ( );
    private int identifierIndex;
    private int priorInputWeightCardinality;
    private int nextInputWeightCardinality;

    public Neuron ( double eta, double alpha, int identifierIndex, int priorInputWeightCardinality, int nextInputWeightCardinality )
    {
        gradient = 0;
        this.identifierIndex = identifierIndex;
        this.priorInputWeightCardinality = priorInputWeightCardinality;
        this.nextInputWeightCardinality = nextInputWeightCardinality;
        this.eta = eta;
        this.alpha = alpha;

        //initialize weights;
        for ( int i = 0; i < nextInputWeightCardinality; i ++ )
        {
            synapses.add ( new Synapse () );
            synapses.get ( i ).setWeight ( new Random ().nextDouble () );
        }
    }

    public double getOutcome ( )
    {
        return outcome;
    }

    public double getGradient ( )
    {
        return gradient;
    }

    public double getLearningRate ( )
    {
        return eta;
    }

    public double getMomentum ( )
    {
        return alpha;
    }

    public double getActivation ( double value )
    {
        return Math.tanh ( value );
    }

    public double getPrimeActivation ( double value )
    {
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }

    public double getDistributedWeightSigma ( Layer subsequentLayer ) 
    {
        double sigma = 0;

        for ( int sLI = 0; sLI < subsequentLayer.size ( ) - 1; sLI ++ ) //sLI -> subsequent layer iterator
            sigma += getSynapses ( ).get ( sLI ).getWeight ( ) * subsequentLayer.get ( sLI ).getGradient ( );
        
        //Note: subsequentLayer.size ( ) - 1 applies, because subsequentLayer size exceeds the number of weights generated for this particular neuron.
        //...Eg, for hidden layer neuron 2, number of weights generated (in constructor) is < nextInputWeightCardinality, or less than 1, starting at 0
        //...so this means 1 weight was generated at 0. So, Going up to < subsequent layer size = 2 (aka going up to index 1) is illegal while accessing
        //...( sLI ).getWeight ( ), because the 1'nth weight simply doesn't exist in this scenario!
        
        return sigma;
    }
    
    public ArrayList <Synapse> getSynapses ( )
    {
        return synapses;
    }
    
    public void setMomentum ( double value )
    {       
        alpha = value;
    }

    public void setGradient ( double value )
    {
        gradient = value;
    }
    
    public void setOutcome ( double value )
    {
        outcome = value;
    }
    
    public void propagateForward ( Layer priorLayer )
    {
        double sigma = 0;

        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ ) //pLI -> prior layer iterator
            sigma += priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).getWeight ( ) * priorLayer.get ( pLI ).getOutcome ( );

        setOutcome ( getActivation ( sigma ) );
    }

    public void computeOutcomeGradient ( double targetValue )
    {
        double delta = targetValue - outcome;
        setGradient ( delta * getPrimeActivation ( outcome ) );
    }


    public void computeHiddenGradient ( Layer subsequentLayer ) 
    {
        setGradient ( getDistributedWeightSigma ( subsequentLayer ) * getPrimeActivation ( outcome ) );
    }

    public void updateWeights ( Layer priorLayer )
    {
        for ( int pLI = 0 ; pLI < priorLayer.size ( ); pLI ++ ) //pLI -> prior layer iterator
        {
            double priorDeltaWeight = priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).getDeltaWeight ( ); //This identifierIndex goes up to two at most, as generated in NeuralNetwork_xOR constructor @lsI, while priorLayer size goes up to three. This is why 1 is subtracted from synonymousLayer size in the inner loop of the weight update section, where updateWeights function is called in the NeuralNetwork_xOR class.


            double synonymousDeltaWeight = eta * gradient * priorLayer.get ( pLI ).getOutcome ( ) + ( alpha * priorDeltaWeight );

            priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).setDeltaWeight ( synonymousDeltaWeight );

            priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).setWeight ( priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).getWeight ( ) + synonymousDeltaWeight );
        }
    }
}
