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
        max_sugar = max(self.model.grid.neighbors_at(("sugar", self.model.grid.find_agent(self)[1]), self.vision))
        candidates = [pos for pos in neighborhood if self.get_sugar(pos) == max_sugar]

        # Narrow down to the nearest ones
        min_dist = min([self.model.grid.distance(
                    self.model.grid.find_agent(self)[1], pos) for pos in candidates])
        final_candidates = [pos for pos in candidates if self.model.grid.distance(
                    self.model.grid.find_agent(self)[1], pos) == min_dist]
        self.model.grid.move_agent(("agents", self.random.choice(final_candidates)), self)

    def eat(self):
        pos = self.model.grid.find_agent(self)[1]
        sugar = self.get_sugar(pos)
        self.sugar = self.sugar - self.metabolism + sugar
        self.model.grid["sugar"][pos] = 0

    def step(self):
        self.move()
        self.eat()
        if self.sugar <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
