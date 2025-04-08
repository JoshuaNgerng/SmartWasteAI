from collector import (
	GarbageCollector, Schedule, EvalSheet,
)
from typing import Callable
from sys import argv
from pathlib import Path
from deap import base, tools, creator, algorithms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import colorsys
import os

def skewed_sample(start, end, scale=5):
	value = int(np.random.exponential(scale)) + start
	return min(value, end)

def get_dist(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
	x1, y1 = pt1; x2, y2 = pt2
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Deap_VRP:
	collector: GarbageCollector
	creator_template: None | Callable[[], any] = None

	def __init__(self, collector: GarbageCollector):
		self.collector = collector
		size = len(collector.locations)
		self.dist_matrix = np.zeros((size, size))
		for i in range(size):
			for j in range(size):
				a = collector.getLoc(i + 1)
				b = collector.getLoc(j + 1)
				self.dist_matrix[i][j] = get_dist(a, b)

	def resigterCreatorTemplate(self, creator: Callable[[], list | Schedule]):
		self.creator_template = creator

	def makeNewCreator(self) -> list | Schedule:
		return self.creator_template()

	def getDist(self, id1, id2):
		return self.dist_matrix[id1 - 1][id2 - 1]

	@staticmethod
	def findDayInSchedule(req: int, schedule: list[int]) -> int:
		day = 1
		for id in schedule:
			if id == 0:
				day += 1
			if req == id:
				return day
		return 0

	@staticmethod
	def getDayIndexesInSchedule(day: int, schedule: list[int]) -> tuple[int, int]:
		day_count = 1; start = 0; end = len(schedule)
		for idx, id in enumerate(schedule):
			if id != 0:
				continue
			day_count += 1
			if day_count == day:
				start = idx + 1
			if day_count == day + 1:
				end = idx
				break
		return (start, end)

	@staticmethod
	def removeDup(schedule: list[int]) -> tuple[list, set]:
		res = []
		seen = set()
		for el in schedule:
			if el == 0:
				res.append(el)
				continue
			if el not in seen:
				seen.add(el)
				res.append(el)
		return (res, seen)

	@staticmethod
	def splitSchedule(schedule: list[int]) -> list[list[int]]:
		buffer = []
		res = []
		for id in schedule:
			if id == 0:
				res.append(buffer)
				buffer = []
				continue
			buffer.append(id)
		res.append(buffer)
		return res

	def evaluateTruckDaySchedule(
			self, day_count, day_schedule: np.ndarray, detail: bool
	) -> EvalSheet:
		class Buffer(EvalSheet):
			dist_check = 0
			capacity = 0
			no_truck = 1
			prev_pos = 1

		buffer = Buffer()
		truck_route = []
		day_route = []
		collector = self.collector
		depo = 1

		def add_truck_route(slots: tuple[int], dist: float):
			buffer.total_truck_dist += dist
			buffer.dist_check += dist
			if detail != True:
				return
			for slot in slots:
				truck_route.append(slot)

		def add_new_truck(slot: int, total: int, new: int):
			nonlocal truck_route
			buffer.no_truck += 1
			buffer.total_truck_dist += total
			buffer.dist_check = new
			if detail != True:
				return
			if len(truck_route) > 0:
				day_route.append(truck_route)
			truck_route = []
			truck_route.append(slot)

		if not day_schedule:
			return EvalSheet()
		for slot in day_schedule:
			req: GarbageCollector.Request = collector.requests[slot - 1]
			new_cap = collector.getCap(slot)
			next = req.location
			next_dist = self.getDist(buffer.prev_pos, next)
			ret_next_dist = self.getDist(next, depo)
			ret_prev_dist = self.getDist(buffer.prev_pos, depo)
			if buffer.capacity + new_cap > collector.truck.capacity:
				if buffer.dist_check + ret_prev_dist + ret_next_dist < collector.truck.max_distance:
					add_truck_route((0, slot), ret_prev_dist + ret_next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity = new_cap
			else:
				if buffer.dist_check + next_dist + ret_next_dist < collector.truck.max_distance:
					add_truck_route((slot,), next_dist)
				else:
					add_new_truck(slot, ret_prev_dist + ret_next_dist, ret_next_dist)
				buffer.capacity += new_cap
			if day_count < req.first:
				buffer.truck_violation += 1
				buffer.penalty += collector.hard_penalty * (req.first - day_count)
			elif day_count > req.last:
				buffer.truck_violation += 1
				buffer.penalty += collector.hard_penalty * (day_count - req.last)
			buffer.prev_pos = next

		buffer.total_truck_dist += self.getDist(buffer.prev_pos, depo)
		buffer.total_truck += buffer.no_truck
		if buffer.max_truck < buffer.no_truck:
			buffer.max_truck = buffer.no_truck
		if detail == True:
			if len(truck_route) > 0:
				day_route.append(truck_route)
		buffer.schedule = day_route
		buffer.__class__ = EvalSheet
		return buffer

	def evaluateTruckSchedule(
			self, schedule: np.ndarray, detail: bool = False
		) -> EvalSheet:
		eval_sheet = EvalSheet()
		day_schedule = self.splitSchedule(schedule)
		for idx, day in enumerate(day_schedule):
			if len(day) == 0:
				if detail == True:
					eval_sheet.schedule.append([])
				continue
			buffer = self.evaluateTruckDaySchedule(idx + 1, day, detail)
			eval_sheet.add(buffer)
		eval_sheet.cal_cost(self.collector)
		return eval_sheet

	def truckScheduleInit(self) -> list[int]:
		added = []
		avaliable = []
		urgent = []
		res = []

		def get_valid_requests(day: int):
			for req in self.collector.requests:
				id = req.id
				if day > req.last or day < req.first:
					continue
				if id in added:
					continue
				if day == req.last:
					urgent.append(id)
				else:
					avaliable.append(id)

		for day in range(1, self.collector.days):
			buffer = []
			get_valid_requests(day)
			if len(avaliable) == 0 and len(urgent) == 0:
				res.append(0)
				continue
			if len(avaliable) > 0:
				no = random.randint(0, len(avaliable))
				buffer = random.sample(avaliable, no)
			buffer.extend(random.sample(urgent, len(urgent)))
			res.extend(buffer)
			added.extend(buffer)
			avaliable.clear()
			urgent.clear()
			res.append(0)
		return res

	def mutateSwapDayOrder(
			self, individual: list[int], indpb: float
		) -> tuple[list[int]]:
		if random.random() > indpb:
			return (individual, )
		day = random.randint(1, self.collector.days)
		start, end = self.getDayIndexesInSchedule(day, individual)
		if end == start:
			return (individual, )
		buffer = random.sample(individual[start:end], end - start)
		individual[start:end] = buffer
		return (individual, )
	
	def mutateExchangeDeliverDay(
			self, individual: list[int], indpb: float
		) -> tuple[list[int]]:
		def find_request(id):
			day = 1
			index = 0
			for idx, slot in enumerate(individual):
				if slot == 0:
					day += 1
				if slot == id:
					index = idx
			return (day, index)

		def change_request_day(id, day, index):
			req: GarbageCollector.Request = self.collector.requests[id - 1]
			new_day = random.choice(range(req.first, req.last))
			if new_day == day and new_day - 1 >= req.first:
				new_day -= 1
			start, end = self.getDayIndexesInSchedule(new_day, individual)
			new_index = start + 1
			if start + 1 < end:
				new_index = random.choice(range(start + 1, end))
			if new_index > index:
				new_index, index = index, new_index
			element = individual.pop(index)
			individual.insert(new_index, element)

		if random.random() > indpb:
			return (individual, )
		requests = self.collector.getReqIds()
		no = skewed_sample(1, len(requests))
		req = random.sample(requests, no)
		for id in req:
			day, index = find_request(id)
			change_request_day(id, day, index)
		return (individual, )

	def crossoverTruck(
			self, parent1: list[int], parent2: list[int]
		) -> tuple[list[int]]:
		def find_breakpoint_index(breakpoint: int):
			index1 = 0
			index2 = 0
			day_count1 = 1
			day_count2 = 1
			for idx, (elem1, elem2) in enumerate(zip(parent1, parent2)):
				if elem1 == 0:
					day_count1 += 1
					if day_count1 == breakpoint:
						index1 = idx
				if elem2 == 0:
					day_count2 += 1
					if day_count2 == breakpoint:
						index2 = idx
			return (index1, index2)

		def repair_offspring(offspring: list):
			def add_missing_slot(lst: list, diff: list, ref: list):
				for slot in diff:
					day = self.findDayInSchedule(slot, ref)
					start, end = self.getDayIndexesInSchedule(day, lst)
					index = start + 1
					if start + 1 < end:
						index = random.randint(index, end)
					lst.insert(index, slot)

			res, seen = self.removeDup(offspring)
			seen.add(0)
			diff1 = [id for id in parent1 if id not in seen]
			diff2 = [id for id in parent2 if id not in seen and id not in diff1]
			add_missing_slot(res, diff1, parent1)
			add_missing_slot(res, diff2, parent2)
			return res

		breakpoint = random.randint(1, self.collector.days)
		index1, index2 = find_breakpoint_index(breakpoint)
		buffer1 = repair_offspring(parent1[:index1] + parent2[index2:])
		buffer2 = repair_offspring(parent2[:index2] + parent1[index1:])
		offspring1 = self.makeNewCreator(); offspring2 = self.makeNewCreator()
		offspring1.extend(buffer1); offspring2.extend(buffer2)
		return (offspring1, offspring2)

def run_deap_loop(
		deap: Deap_VRP, fname_input: str,
		size: int = 200, max_iter_truck: int = 500,
		p_cx: float = 0.9, p_mut: float = 0.8, indpb: float = 0.01
	) -> list[int]:
	def mutExplore(individual: list[int], indpb: float) -> tuple[list[int]]:
		mut_func = [deap.mutateSwapDayOrder, deap.mutateExchangeDeliverDay]
		rand_func = random.choice(mut_func)
		return rand_func(individual, indpb)

	def Truckfitness(individual: list[int]) -> tuple[int]:
		buffer = deap.evaluateTruckSchedule(individual)
		return (buffer.total_cost, )

	def sel_elite_tourn(individuals, size, tournsize):
		k_elite = int(size * 0.1)
		k = size - k_elite
		return (tools.selBest(individuals, k_elite) +
				tools.selTournament(individuals, k, tournsize))

	def remove_outliers(data: np) -> np:
		Q1 = np.percentile(data, 25)
		Q3 = np.percentile(data, 75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR
		filtered_data = [
			ele[0] for ele in data if ele[0] >= lower_bound and ele[0] <= upper_bound
		]
		return np.array(filtered_data)

	def average_without_outliers(data) -> float:
		return remove_outliers(data).mean()

	def std_without_outliers(data) -> float:
		buffer = remove_outliers(data)
		return np.std(buffer) if len(buffer) > 1 else 0 

	def save_plt(logbook, fname: str, label: str):
		output_fname = fname.replace(".txt", f"_{label}.jpeg")
		plt.plot(logbook.select("min"), color='red')
		plt.plot(logbook.select("avg"), color='green')
		plt.xlabel('Generations')
		plt.ylabel('best/average fitness per population')
		plt.savefig(output_fname, format="jpeg")
		plt.clf()
		output_fname = fname.replace(".txt", f"_{label}_std.jpeg")
		plt.plot(logbook.select("std"), color='blue')
		plt.xlabel('Generations')
		plt.ylabel('std of population')
		plt.savefig(output_fname, format="jpeg")
		plt.clf()

	toolbox = base.Toolbox()
	creator.create("fitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("individual", list, fitness=creator.fitnessMin)
	toolbox.register(
		"individualInit", tools.initIterate,
		creator.individual, deap.truckScheduleInit
	)
	toolbox.register(
		"populationInit", tools.initRepeat,
		list, toolbox.individualInit
	)
	deap.resigterCreatorTemplate(creator.individual)
	toolbox.register("select", sel_elite_tourn, tournsize=2)
	toolbox.register("evaluate", Truckfitness)
	toolbox.register("mate", deap.crossoverTruck)
	toolbox.register("mutate", mutExplore, indpb=indpb)
	stats = tools.Statistics(lambda ind:ind.fitness.values)
	stats.register("min", np.min)
	stats.register("avg", average_without_outliers)
	stats.register("std", std_without_outliers)
	hof = tools.HallOfFame(5)
	starting_pop = toolbox.populationInit(size)
	print("Finding best truck route")
	final_pop, logbook = algorithms.eaSimple(
		starting_pop, toolbox, p_cx, p_mut,
		max_iter_truck, stats, hof, False
	)
	print("Found best truck route")
	save_plt(logbook, fname_input, "truck")
	return hof[0]

def generate_vrp_map(
		fname: str, collector: GarbageCollector, schedule: Schedule, max_truck: int
	):
	# Function to generate colors dynamically using HSL space
	def generate_colors(n):
		# We will generate n distinct colors by modifying the hue
		colors = []
		for i in range(n):
			# Vary the hue between 0 and 1 (360 degrees on the color wheel)
			hue = i / n  # This will give us a distinct hue for each color
			# Convert HSL to RGB, keeping saturation and lightness fixed
			rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)  # Saturation = 0.5, Lightness = 0.7
			colors.append(rgb)
		return colors
	
	def plot_points():
		plt.figure(figsize=(10, 8))
		for loc_id, x, y in collector.locations:
			plt.scatter(
				x, y, color='red', s=50, label=f'Location {loc_id}'
			)
			plt.text(x + 5, y, f'{loc_id}', fontsize=12, color='black')

	# Generate a list of colors for all trucks
	colors = generate_colors(max_truck)
	
	
	# Plot all the locations first

	# Plot the routes for each truck
	for idx, day in enumerate(schedule):
		plot_points()
		for idx_, route in enumerate(day):
			truck_color = colors[idx_ - 1]  # Get the color for the current truck
			for i in range(len(route)):
				# Plot the route line
				start_loc = route[i]
				end_loc = route[(i+1) % len(route)]
				if start_loc == 0: start_loc = 1
				if end_loc == 0: end_loc = 1
				# Get coordinates of the starting and ending location
				start_x, start_y = collector.getReqLoc(start_loc)
				end_x, end_y = collector.getReqLoc(end_loc)
				# Plot a line between the start and end locations
				plt.plot(
					[start_x, end_x], [start_y, end_y], color=truck_color,
					linestyle='-', linewidth=2
				)
		output_fname = fname.replace('.txt', f'_{idx + 1}_route.png')
		print(f"getting route {output_fname}")
		# Add labels and title
		plt.title(f'Day {idx + 1} Routes')
		plt.xlabel('X Coordinate')
		plt.ylabel('Y Coordinate')
		plt.grid(True)
		plt.savefig(output_fname, format='png')
		plt.clf()

def main(input_fname: str, seed: int = 200):
	output_fname = input_fname.replace(".txt", "_out.txt")
	collector = GarbageCollector(input_fname, seed=seed)
	deap = Deap_VRP(collector)
	matplotlib.use('Agg')
	dir_name = Path(input_fname.replace(".txt", ""))
	dir_name.mkdir(parents=True, exist_ok=True)
	os.chdir(dir_name)
	size = 300
	max_iter_truck = 500
	res = run_deap_loop(
		deap, input_fname,
		size=size, max_iter_truck=max_iter_truck
	)
	eval = deap.evaluateTruckSchedule(res, True)
	schedule = Schedule(eval.schedule)
	with open(output_fname, 'w') as f:
		f.write(eval.format_str())
		f.write(str(schedule))
	generate_vrp_map(input_fname, collector, schedule, eval.max_truck)

if __name__ == "__main__":
	if len(argv) != 2 and len(argv) != 3:
		print("Please give argument (input file name) to the script")
	else:
		seed = 200
		if len(argv) == 3:
			seed = int(argv[2])
		main(str(argv[1]), seed)
