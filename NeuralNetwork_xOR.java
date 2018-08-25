//author: Jordan Micah Bennett
import java.util.ArrayList;

public class NeuralNetwork_xOR
{
    private double eta = 0.2;
    private double alpha = 0.5;
    private Topology topology;
    private Layers layers;
    private double netError;

    public NeuralNetwork_xOR ( )
    {
        layers = new Layers ( );
        topology = new Topology ( "2,2,1" );

        //lsI -> layers iterator (plural layers)
        for ( int lsI = 0; lsI < topology.size ( ); lsI ++ )
        {
                layers.add ( new Layer ( ) ); //populate each layer holder with an empty but defined/instantiated layer instance
           
                //lI -> layer iterator (singular layer)
                for ( int lI = 0; lI <= topology.get ( lsI ); lI ++ ) //<= implies generation of an extra bias neuron, for variability of activations beyond origin
                {
                    int nextInputWeightCardinality = lsI + 1 < topology.size ( ) ? topology.get ( lsI + 1 ) : 0; //Get number of weights in next layer from current neuron. Used to initialize weights in neuron object.
                    int priorInputWeightCardinality = lsI - 1 > 0 ? topology.get ( lsI - 1 ) : 0;
                    layers.get ( lsI ).add ( new Neuron ( eta, alpha, lI, priorInputWeightCardinality, nextInputWeightCardinality ) );                          
                }
        
                for ( int lI = 0; lI <= layers.get ( lsI ).size ( ); lI ++ ) //<= implies generation of an extra bias neuron, for variability of activations beyond origin
                {
                    Neuron lastNeuronInLayer = layers.get ( lsI ).get ( layers.get ( lsI ).size ( ) - 1 );
                    
                    lastNeuronInLayer.setOutcome ( 1.0 );     //bias neuron 
                }
        }
    }


    public void forwardPropagation ( ArrayList <Double> inputValues )
    {
        //populate input layer or first layer of neural network with input values
        for ( int iVI = 0; iVI < inputValues.size ( ); iVI ++ ) // iVI -> input values iterator
            layers.get ( 0 ).get ( iVI ).setOutcome ( inputValues.get ( iVI ) );

        /*
         * populate other layers, or setup other layers by setting up each layer neuron's outcome wrt to each prior layer,
         * including first layer above
         */
        for ( int lsI = 1; lsI < topology.size ( ); lsI ++ ) // lsI -> layers iterator (plural layers)
        {
            Layer priorLayer = layers.get ( lsI - 1 );
            
            for ( int lI = 0; lI < topology.get ( lsI ); lI ++ )  // lI -> layer iterator (singular layer)
            {
                layers.get ( lsI ).get ( lI ).propagateForward ( priorLayer );
            }
        }
    }

    public void backwardPropagation ( ArrayList <Double> targetValues )
    {
        Layer outputLayer = layers.get ( layers.size ( ) - 1 );

        double sigma = 0;
        
        for ( int tVI = 0; tVI < targetValues.size ( ); tVI ++ )
            sigma += Math.pow ( targetValues.get ( tVI ) - outputLayer.get ( tVI ).getOutcome ( ), 2 );

        netError = sigma / outputLayer.size ( );
        //the netError is used to show the accuracy of the model, and are not used to generate actual training process

        //outcome gradient
        for ( int tVI = 0; tVI < targetValues.size ( ); tVI ++ )
            outputLayer.get ( tVI ).computeOutcomeGradient ( targetValues.get ( tVI ) );


        //hidden gradient - Update gradients from hidden layer or layer before last, up until first, excluding the first. (Aka update middle layer gradients)
        for ( int lsI = layers.size ( ) - 2; lsI > 0; lsI -- )
        {
            Layer subsequentLayer = layers.get ( lsI + 1 );
            Layer hiddenLayer = layers.get ( lsI );

            for ( int lI = 0; lI < hiddenLayer.size ( ); lI ++ )
                hiddenLayer.get ( lI ).computeHiddenGradient ( subsequentLayer );
        }


        //weight update - Update weights from last layer up to/excluding the first. (Aka update all weights in model)
        //For xOr problem space, the neural network structure has two sets of weights; i.e. from last layer to hidden/middle, and from middle to first!
        for ( int lsI = layers.size ( ) - 1; lsI > 0; lsI -- )
        {
            Layer priorLayer = layers.get ( lsI - 1 );
            Layer synonymousLayer = layers.get ( lsI );

            for ( int lI = 0; lI < synonymousLayer.size ( ) - 1; lI ++ )
                synonymousLayer.get ( lI ).updateWeights ( priorLayer );
        }       
    }

    public ArrayList <Double> getOutcomes ( )
    {
        ArrayList <Double> returnValues = new ArrayList <Double> ( );
        
        Layer outputLayer = layers.get ( layers.size ( ) - 1 );

        //lI -> layer iterator
        for ( int lI = 0; lI < outputLayer.size ( ) - 1; lI ++ ) //return all outcomes, except the last bias neuron
            returnValues.add ( outputLayer.get ( lI ).getOutcome ( ) );

        return returnValues;
    }
    
    public double getAccuracy ( )
    {
        return netError;
    }
}
