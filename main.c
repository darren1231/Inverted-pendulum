#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <process.h>

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5		  /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02		  /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333
#define sqr(x) ( (x) * (x) )
#define W_INIT 0.0
#define NUM_BOXES 162
static int RND_SEED = 8800;            //0.4   0.85  2.4     0.4 0.6  x2  F2=-10   0.4 0.8 F2 ¥¿±`    0.55 0.8   x=1 
static double ALPHA = 0.61;              /* learning rate parameter */
static double BETA = 0.01;               /* magnitude of noise added to choice */
static double GAMMA = 0.87;            /* discount factor for future reinf */
static double q_val[NUM_BOXES][5];      /* state-action values */
static first_time = 1;
static int cur_action, prev_action;
static int cur_state, prev_state;

int get_action(double,double,double,double,double);
void cart_pole(int,double*,double*,double*,double*);
double rnd(double, double);
int get_box(double, double, double, double);  /*state*/
void reset_controller(void);    /* reset state/action before new trial */

int main()
{
	int action,box,i;
	long success,trial;
	double x, x_dot, theta, theta_dot,reinf,predicted_value;
	FILE *fptr;
	FILE *fptr1;
	fptr=fopen("rand_restart.txt","w");
	fptr1=fopen("output.csv","w");
	x=x_dot=theta=theta_dot=rnd(-BETA,BETA);
    double angle;
	success=0;
	trial=1;
	reinf=0.0;
	double force;
	double j,k;
	double best_ALPHA=0;
	double best_GAMMA=0;
    
    while (success<1000000)    /* If the pole doesn't fall during 1-million trials,assume it succcess.*/
	{
          
          //getchar();
		action=get_action(x,x_dot,theta,theta_dot,reinf);
		cart_pole(action,&x,&x_dot,&theta,&theta_dot);
		
		//printf("%d")
        if(action==0)
        force=10;
        else if(action==1)
        force=5;
        else if(action==2)
        force=0;
        else if(action==3)
        force=-5;
        else
        force=-10;
		
		
		
        fprintf(fptr1,"%.2f,%.2f,%.2f,%.2f,%f\n",x,theta,x_dot,theta_dot,force);
		angle=theta*180/3.1415926;
		//printf("x%.2f,angle%.2f,%.2f,%.2f,%d\n",x,angle,x_dot,theta_dot,action);
		
		box=get_box(x,x_dot,theta,theta_dot);
		if (box==-1)
		{	reinf=-1.0;
			predicted_value = 0.0;
	        q_val[prev_state][prev_action]
				  += ALPHA * (reinf + GAMMA * predicted_value - q_val[prev_state][prev_action]);
			reset_controller();
			x=x_dot=theta=theta_dot=rnd(-BETA,BETA);
			trial++;
			//printf("At %d success ,try %d trials\n",success,trial);
			printf("At trial %d : success--->%d (mean last how long)\n",trial,success);
			fprintf(fptr,"trials%d\t success%d\n",trial,success);
			success=0;
		}else{
			  success++;
			  reinf=0.0;
			  /*if(success>1000000-2)
			  {
              printf("asfasdfasdf");        
              break;
              }*/
			  
			}
	}
		printf("If success > 1000000 \n Success at %d trials \n",trial);
        for (i=0;i<NUM_BOXES;i++)
		fprintf(fptr,"%g %g\n",q_val[i][0],q_val[i][1]);
	fclose(fptr);
    fclose(fptr1);
    
 
 



system("pause");

}

/*----------------------------------------------------------------------
   cart_pole:  Takes an action (0 or 1) and the current values of the
 four state variables and updates their values by estimating the state
 TAU seconds later.
----------------------------------------------------------------------*/

void cart_pole(int action,double *x,double *x_dot,double *theta,double *theta_dot)
{
    double xacc,thetaacc,force,costheta,sintheta,temp;

    if(action==0)
    force=10;
    else if(action==1)
    force=5;
    else if(action==2)
    force=0;
    else if(action==3)
    force=-5;
    else
    force=-10;
    
    costheta = cos(*theta);
    sintheta = sin(*theta);

    temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
		         / TOTAL_MASS;

    thetaacc = (GRAVITY * sintheta - costheta* temp)
	       / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
                                              / TOTAL_MASS));

    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

/*** Update the four state variables, using Euler's method. ***/

    *x  += TAU * *x_dot;
    *x_dot += TAU * xacc;
    *theta += TAU * *theta_dot;
    *theta_dot += TAU * thetaacc;
}

/*** Q-learning part ***/
/*----------------------------------------------------------------------------+
|   get_action : returns either 0 or 1 as action choice;                      |
|                accepts five "black box" inputs, the first four of which are |
|                   system variables, and the last is a reinforcement signal. |
|                Note that reinf is the result of the previous state and      |
|                   action.                                                   |
+----------------------------------------------------------------------------*/

int get_action(double x,            /* system variables == state information */
               double x_dot, 
               double theta, 
               double theta_dot, 
               double reinf)        /* reinforcement signal */
{
   int i,j;
   double predicted_value;          /* max_{b} Q(t, ss, b) */
   if (first_time) {
      first_time = 0;
      reset_controller();   /* set state and action to null values */

      for (i = 0; i < NUM_BOXES; i++)
         for (j = 0; j < 5; j++)
            q_val[i][j] = W_INIT;
  
      printf("... setting learning parameter ALPHA to %.4f.\n", ALPHA);
      printf("... setting noise parameter BETA to %.4f.\n", BETA);
      printf("... setting discount parameter GAMMA to %.4f.\n", GAMMA);
      printf("... random RND_SEED is %d.\n", RND_SEED);
      srand(RND_SEED);    /* initialize random number generator */
   }

   prev_state = cur_state;
   prev_action = cur_action;
   cur_state = get_box(x, x_dot, theta, theta_dot);
   
   double max_predicted_q_val;
   max_predicted_q_val=q_val[cur_state][1];
   predicted_value=q_val[cur_state][1];
   
   //for (i = 0; i < 5; i++)
  // printf("q_val[cur_state][1]  %f\n",q_val[cur_state][i]);
   
   
   if (prev_action != -1)  /* Update, but not for first action in trial */
   {
      if (cur_state == -1)
         /* failure state has Q-value of 0, since the value won't be updated */
         predicted_value = 0.0;
      else  
      {
           for (i = 0; i < 5; i++)    
           if(q_val[cur_state][i] >= max_predicted_q_val)
           {
           max_predicted_q_val= q_val[cur_state][i];                      
           predicted_value = q_val[cur_state][i];
           
          // printf("change predicted_value%d  %f\n",i,predicted_value);
           }
      }
    // printf("predicted_value  %f\n",predicted_value);
      
      
      
      q_val[prev_state][prev_action]
        += ALPHA * (reinf + GAMMA * predicted_value 
                          - q_val[prev_state][prev_action]);
   }
   //for (i = 0; i < 5; i++)    
   //printf("q_val[cur_state][i]  %f\n",q_val[cur_state][i]);
   
   /* Now determine best action */
   double max_q_val;
   max_q_val=q_val[cur_state][0];
   cur_action = 0;
   
   for (i = 0; i < 5; i++)
   if (q_val[cur_state][i]- rnd(-BETA, BETA)  >= max_q_val)//
   {
      max_q_val=q_val[cur_state][i];
      cur_action = i;
   }
  
      
  // printf("cur_action%d %f\n",cur_action,rnd(-BETA, BETA));
   return cur_action;
}


double rnd(double low_bound, double hi_bound)
/* rnd scales the output of the system random function to the
 * range [low_bound, hi_bound].
 */
{
   double highest = 99999;
   return (rand() / highest) * (hi_bound - low_bound) + low_bound;
}


void reset_controller(void)
{
   cur_state = prev_state = 0;
   cur_action = prev_action = -1;   /* "null" action value */
}


/* The following routine was written by Rich Sutton and Chuck Anderson,
      with translation from FORTRAN to C by Claude Sammut  */

/*----------------------------------------------------------------------
   get_box:  Given the current state, returns a number from 0 to 161
  designating the region of the state space encompassing the current state.
  Returns a value of -1 if a failure state is encountered.
----------------------------------------------------------------------*/

#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifty_degrees 0.87266

int get_box(double x, double x_dot, double theta, double theta_dot)
{
  int box=0;

  if (x < -1.0 ||
      x > 1.0 ||
      theta < -twelve_degrees ||
      theta > twelve_degrees)          return(-1); /* to signal failure */

  if (x < -0.8)  		       box = 0;
  else if (x < 0.8)     	       box = 1;
  else		    	               box = 2;

  if (x_dot < -0.5) 		       ;
  else if (x_dot < 0.5)                box += 3;
  else 			               box += 6;

  if (theta < -six_degrees) 	       ;
  else if (theta < -one_degree)        box += 9;
  else if (theta < 0) 		       box += 18;
  else if (theta < one_degree) 	       box += 27;
  else if (theta < six_degrees)        box += 36;
  else	    			       box += 45;

  if (theta_dot < -fifty_degrees) 	;
  else if (theta_dot < fifty_degrees)  box += 54;
  else                                 box += 108;

  return(box);
}
