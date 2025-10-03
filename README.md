[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lm-7f901)
# README 

Sometimes, I leave my beverage package in the hot car, so I end up with aluminum-clad cylinders of mostly water with, what I suspect, is a pretty uniform temperature profile that peaks at $T_{hot car} = 95~{}^{\text{o}$F.  When I found I had just done it again most recently, I naturally wondered how I might remedy the solution in an optimal matter, right after I, surely suboptimally, spun the can in an icebath until I got bored (say, 30 seconds), and then repeated the process twice more (with 5-minute breaks after each spin sesh).  
 
Now, were the first can not so pote..ntially the best ever, I'd have sat down and modeled all $3(30+5(60)) \approx 1000$ s of the temperature shift to(ward?) $45~{}^{\text{o}}$F before enjoying.  So, I called an audible, and this is the result. 

Your Tasks:

 - Model the can and bath using an appropriate form of the transient heat 
   conduction equation.  Of critical importance is the quality of your 
   assumptions.

 - Implement a numerical solution for this model that provides you an estimate
   for how quickly this can can be lowered to an enjoyable temperature.

 - Provide a lower and upper bound on your estimated time that represents the 
   95% CI.


Deliverable: self contained Python(?) solution with adequate commenting (via docstrings) and testing (via `unittest`).

