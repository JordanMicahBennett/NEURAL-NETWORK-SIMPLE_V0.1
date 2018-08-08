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
            double priorDeltaWeight = priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).getDeltaWeight ( );

            double synonymousDeltaWeight = eta * gradient * priorLayer.get ( pLI ).getOutcome ( ) + ( alpha * priorDeltaWeight );

            priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).setDeltaWeight ( synonymousDeltaWeight );

            priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).setWeight ( priorLayer.get ( pLI ).getSynapses ( ).get ( identifierIndex ).getWeight ( ) + synonymousDeltaWeight );
        }
    }
}