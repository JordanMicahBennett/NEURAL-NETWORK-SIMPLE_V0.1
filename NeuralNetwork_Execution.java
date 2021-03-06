  /* 
  * Author: Jordan Micah Bennett
  * Why?: This is practice writing of basic neural network. I do this at the end of each year, sometimes every 6 months, for practice of fundamentals.
  * Date: August 7, 2018 to August 8, 2018. (1 hour used each of the two days)
  * Resource: Ncb i7 laptop
  * 
  * The problem space is X-OR inputs. So the model does xor input prediction.
  * Given two numbers in X-OR space, the model will try to guess the output.
  * 
  * For:
  * a) Input=(0,0) output should be 0
  * b) Input=(1,0) output should be 1
  * c) Input=(0,1) output should be 1
  * d) Input=(1,1) output should be 0
  * 
  */
 
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetwork_Execution
{
    public static void main ( String [ ] arguments )
    {
        ArrayList <String> problemSpace = new ArrayList <String> ( ); 
       
        
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );        
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "0,1:1" );
        problemSpace.add ( "0,0:0" );
        problemSpace.add ( "1,1:0" );
        problemSpace.add ( "1,0:1" );   
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	problemSpace.add ( "1,1:0" );
	problemSpace.add ( "1,0:1" );
	problemSpace.add ( "0,1:1" );
	problemSpace.add ( "0,0:0" );
	
	NeuralNetwork_xOR neuralNetwork_xOR = new NeuralNetwork_xOR ( );
	System.out.println ( "Begin processing " + problemSpace.size ( ) + " inputs..." );
	
	//Run training - Extract training data, and run forward and backrward propagation
	//Each step in loop consumes two inputs and an expected outcome
	for ( int pSI = 0; pSI < problemSpace.size ( ); pSI ++ ) 
	{
	    String [ ] trainingData = problemSpace.get ( pSI ).split ( ":" );
	    
	    ArrayList <Double> inputValues = new ArrayList <Double> ( );
	    
	    String [ ] inputSpace = trainingData [ 0 ].split ( "," );
	    
	    inputValues.add ( Double.parseDouble ( inputSpace [ 0 ] ) );
	    inputValues.add ( Double.parseDouble ( inputSpace [ 1 ] ) );
	    
	    neuralNetwork_xOR.forwardPropagation ( inputValues ); //forward propagate on input signals
	    
	    ArrayList <Double> targetValues = new ArrayList <Double> ( );
	    
	    targetValues.add ( Double.parseDouble ( trainingData [ 1 ] ) );
	    
	    neuralNetwork_xOR.backwardPropagation ( targetValues ); //backward propagate on target data
	}
	
	System.out.println ( "Training Complete.\nOverall Error Rate : " + neuralNetwork_xOR.getAccuracy ( ) + "\n\n" );
	
	
	renderMenu ( neuralNetwork_xOR );
    }

    
    public static void renderMenu ( NeuralNetwork_xOR neuralNetwork_xOR ) 
    {
        System.out.println ( "-------------------------------" );
	System.out.println ( "Test the neural network (aka supply unsupervised input)" );
	System.out.println ( "1. Extract guess for Input = (1,1) " );
	System.out.println ( "2. Extract guess for Input = (1,0) " );
	System.out.println ( "3. Extract guess for Input = (0,1) " );
	System.out.println ( "4. Extract guess for Input = (0,0) " );
	System.out.println ( "5. Exit" );
	Scanner scanner = new Scanner ( System.in );
	int option = Integer.parseInt ( scanner.nextLine ( ) );
	
	switch ( option )
	{
	    case 1:
	    {
	        ArrayList <Double> inputValues = new ArrayList <Double> ( );
	        inputValues.add ( 1.0 );
	        inputValues.add ( 1.0 );
	        neuralNetwork_xOR.forwardPropagation ( inputValues );
	        System.out.println ( "\n\nGuess : " + neuralNetwork_xOR.getOutcomes ( ).get ( 0 ) + " (Correct Value is 0 for [1,1]) \n Press return to continue\n\n" );
	        scanner.nextLine ( );
	        System.out.println ( "\f" );
	        renderMenu ( neuralNetwork_xOR );
	    }
	    break;
	   
	    case 2:
	    {
	        ArrayList <Double> inputValues = new ArrayList <Double> ( );
	        inputValues.add ( 1.0 );
	        inputValues.add ( 0.0 );
	        neuralNetwork_xOR.forwardPropagation ( inputValues );
	        System.out.println ( "\n\nGuess : " + neuralNetwork_xOR.getOutcomes ( ).get ( 0 ) + " (Correct Value is 1 for [1,0]) \n Press return to continue\n\n" );
	        scanner.nextLine ( );
	        System.out.println ( "\f" );
	        renderMenu ( neuralNetwork_xOR );
	    }
	    break;
	    
	    case 3:
	    {
	        ArrayList <Double> inputValues = new ArrayList <Double> ( );
	        inputValues.add ( 0.0 );
	        inputValues.add ( 1.0 );
	        neuralNetwork_xOR.forwardPropagation ( inputValues );
	        System.out.println ( "\n\nGuess : " + neuralNetwork_xOR.getOutcomes ( ).get ( 0 ) + "(Correct Value is 1 for [0,1]) \n Press return to continue\n\n" );
	        scanner.nextLine ( );
	        System.out.println ( "\f" );
	        renderMenu ( neuralNetwork_xOR );
	    }
	    break;
	    
	    case 4:
	    {
	        ArrayList <Double> inputValues = new ArrayList <Double> ( );
	        inputValues.add ( 0.0 );
	        inputValues.add ( 0.0 );
	        neuralNetwork_xOR.forwardPropagation ( inputValues );
	        System.out.println ( "\n\nGuess : " + neuralNetwork_xOR.getOutcomes ( ).get ( 0 ) + "(Correct Value is 0 for [0,0]) \n Press return to continue\n\n" );
	        scanner.nextLine ( );
	        System.out.println ( "\f" );
	        renderMenu ( neuralNetwork_xOR );
	    }
	    break;    
	    
	    
	    case 5:
	    {
	        System.exit ( 0 );
	    }
	    break;
        } 
    }
}
