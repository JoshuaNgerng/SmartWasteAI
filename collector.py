import random
import math
from typing import Iterator, TypeVar, NamedTuple, Optional
from typing_extensions import Self

class GarbageCollector:
	class Truck(NamedTuple):
		capacity: int
		max_distance: int
		distance_cost: int
		day_cost: int
		cost: int
	class Location(NamedTuple):
		id: int
		x: int
		y: int
	class Request(NamedTuple):
		id: int
		location: int
		first: int
		last: int
		garbage: int

	days: int
	truck: Truck
	locations: list[Location]
	requests: list[Request]
	hard_penalty: int

	def __init__(
			self, fname: str,
			hard_penalty: int | None = None, seed: int | None = None
		):
		self.machines = []
		self.locations = []
		self.requests = []
		T = TypeVar('T', bound=NamedTuple)
		truck = {
			"truck_capacity": 0, "truck_max_distance": 0,
			"truck_distance_cost": 0, "truck_day_cost": 0, "truck_cost": 0
		}
		mapping = {
			"locations": (self.locations, self.Location),
			"requests": (self.requests, self.Request)
		}
		def sort(src: list[T]) -> list[T]:
			return sorted(src, key=lambda src: src.id)

		def parse_map(
				dst: list[T], ref: list[T],
				start: int, no: int, lines: list[str]
			):
			"""
			Generic function that parses a space-separated line
			into a namedtuple based on the provided template.
			"""
			count = 0
			for line in lines[start:]:
				if count == no:
					break
				values = list(map(int, line.split()))
				dst.append(ref(*values))
				count += 1

		with open(fname, 'r') as f:
			lines = f.readlines()
			i = 0
			while i < len(lines):
				line = lines[i].strip()
				i += 1
				if len(line) == 0:
					continue
				key, no = line.split("=")
				key = key.strip().lower()
				no = int(no.strip())
				if key == "days":
					self.days = int(no)
				elif key in truck:
					truck[key] = no
				elif key in mapping:
					dst, ref = mapping[key]
					parse_map(dst, ref, i, no, lines)
					i += no
		self.locations = sort(self.locations)
		self.requests = sort(self.requests)
		self.truck = self.Truck(*truck.values())
		if hard_penalty is None:
			hard_penalty = max(truck.values()) ** 2
		self.hard_penalty = hard_penalty
		if isinstance(seed, int):
			random.seed(seed)

	def __str__(self) -> str:
		s = f"DAYS: {self.days}\n"
		s += str(self.truck) + '\n'
		s += "locations\n"
		for loc in self.locations:
			s += str(loc) + '\n'
		s += "requests\n"
		for req in self.requests:
			s += str(req) + '\n'
		return s

	def getLoc(self, id: int) -> tuple[int]:
		return (self.locations[id - 1].x, self.locations[id - 1].y)

	def getReqLoc(self, id: int) -> tuple[int]:
		return self.getLoc(self.requests[id - 1].location)

	def getReqLocId(self, id: int) -> int:
		return self.requests[id - 1].location

	def getCap(self, id: int) -> int:
		return self.requests[id - 1].garbage

	def getReqIds(self) -> range:
		return self.getRange(self.requests)

	@staticmethod
	def getRange(src_list: list) -> range:
		return range(src_list[0].id, src_list[-1].id + 1)

	@staticmethod
	def getDist(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
		x1, y1 = pt1; x2, y2 = pt2
		return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class DaySchedule:
	truck: list[list[int]] = []
	def __init__(self, truck: list[list[int]]):
		self.truck = truck
	def __str__(self) -> str:
		s = f"NUMBER_OF_TRUCKS={len(self.truck)}\n"
		for idx, route in enumerate(self.truck):
			s += f'{idx + 1} -> [{" ".join(map(str, route))}]\n'
		return s
	def __iter__(self) -> Iterator[list[int]]:
		for route in self.truck:
			yield route

class Schedule:
	day_schedules: list[DaySchedule] = []
	def __init__(self, truck_part: list[list[list[int]]]):
		self.day_schedules = [
			DaySchedule(truck_day)
			for truck_day in truck_part
		]
	def __str__(self) -> str:
		s = "SCHEDULE DETAILS\n"
		for day_count, day_schedule in enumerate(self.day_schedules):
			s += f"DAY = {day_count + 1}\n{day_schedule}"
		return s
	def __iter__(self) -> Iterator[DaySchedule]:
		for day_schedule in self.day_schedules:
			yield day_schedule
	def __getitem__(self, idx: int) -> DaySchedule:
		return self.day_schedules[idx]
	def __len__(self) -> int:
		return len(self.day_schedules)
	def add(self, other: DaySchedule):
		self.day_schedules.append(other)

class EvalSheet:
	total_truck_dist: float = 0.0
	max_truck: int = 0
	total_truck: int = 0
	truck_violation: int = 0
	total_cost: int = 0
	penalty: int = 0
	schedule: list[list[list[int]]] = []
	def __str__(self) -> str:
		s = str(self.__class__.__name__) + '\n'
		s += '\n'.join(f'{key}={value}' for key, value in self.__dict__.items())
		return s
	def add_stat(self, other: Self):
		self.total_truck_dist += other.total_truck_dist
		self.total_truck += other.total_truck
		if self.max_truck < other.max_truck:
			self.max_truck = other.max_truck
		self.truck_violation += other.truck_violation
		self.penalty += other.penalty
		self.total_cost += other.total_cost
	def add(self, other: Self):
		self.add_stat(other)
		if len(other.schedule) > 0:
			self.schedule.append(other.schedule)
	def extend(self, other: Self):
		self.schedule.extend(other.schedule)
	def cal_cost(self, collector: GarbageCollector) -> int:
		self.total_cost = self.total_truck_dist * collector.truck.distance_cost + \
			self.total_truck * collector.truck.day_cost + \
			self.max_truck * collector.truck.cost + self.penalty
		self.total_cost = int(self.total_cost)
		return self.total_cost
	def format_str(self) -> str:
		s = f'TRUCK_DISTANCE={self.total_truck_dist}\n'
		s += f'NUMBER_OF_TRUCK_DAYS={self.total_truck}\n'
		s += f'NUMBER_OF_TRUCK_USED={self.max_truck}\n'
		s += f'TOTAL_COST={self.total_cost}\n\n'
		return s

def test():
	collector = GarbageCollector("test_1.txt")
	for id, x, y in collector.locations:
		print(f'{id} {x} {y}')

if __name__ == "__main__":
	test()