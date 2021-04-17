from Solvers.AbstractSolver import AbstractSolver

class SAC(AbstractSolver):
	def __init__(self, env, options):
		super.__init__(env, options)

	def train_episode(self):
		s = env.reset()

		done = False
		while True:#while not done:
			s_p, r, done, _ = env.step(env.action_space.sample())
			s = s_p