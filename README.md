simulate a theoretical optimal route for garbage truck in urban areas
the input have time window for garbage collection, capicity and distance limit
it output the schedule in a text file in its directory named after the input file
there are many pics to visualize the route of each day
and also the performance of the algo over each generation
usuage

python deap_VRP.py "input filename"


input file format
days in the schedule
truck info etc distance cost
location -> id, x, y
requests -> id, location id, first day, last day, capacity
