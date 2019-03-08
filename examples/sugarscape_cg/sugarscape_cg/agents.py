from mesa import Agent

class SsAgent(Agent):
    id_counter = 0

    def __init__(self, model, sugar=0, metabolism=0, vision=0):
        super().__init__(SsAgent.id_counter, model)
        SsAgent.id_counter += 1
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision

    def get_sugar(self, pos):
        return self.model.grid[("sugar", pos)]

    def is_occupied(self, pos):
        return self.model.grid.count_agents_at(pos)

    def move(self):
        # Get neighborhood within vision
        neighborhood = self.model.grid.neighborhood_of(self, self.vision)

        # Look for location with the most sugar
        max_sugar = max(self.model.grid.neighbors_of(("sugar", self.model.grid.find_agent(self)), self.vision))
        candidates = [pos for pos in neighborhood if self.get_sugar(pos) == max_sugar]

        # Narrow down to the nearest ones
        min_dist = min([self.model.grid["agent"].metric.distance(
                    self.model.grid.find_agent(self), pos) for pos in candidates])


        final_candidates = [pos for pos in candidates if self.model.grid["agent"].metric.distance(
                    self.model.grid.find_agent(self), pos) == mind_dist]

        self.random.shuffle(final_candidates)
        self.model.grid.move_agent(final_candidates[0], self)

    def eat(self):
        sugar_patch = self.get_sugar(self.pos)
        self.sugar = self.sugar - self.metabolism + sugar_patch.amount
        sugar_patch.amount = 0

    def step(self):
        self.move()
        self.eat()
        if self.sugar <= 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
