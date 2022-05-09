import numpy as np

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

import pdist
from scipy.optimize import fsolve
    
dtype = np.double

class Reaction:
    """ Reaction base class 

        Every reaction has the following parameters:

        - rate: float
            Base rate of the reaction

        - products: array or None (default: None)
            List of products created during the reaction. Entries must be of the following forms:
            * spec: int 
                Denotes one reactant of the given species 
            * (b: float, spec: int)
                Denotes a burst of the given species following the geometric distribution with mean b
                (by convention the geometric distribution starts at 0).

    """

    def __init__(self, rate, products = None):
        self.rate = rate
        self.products = products

class GenReaction(Reaction):
    """
        The GenReaction class represents reactions without educts.
    """

    def __init__(self, rate, products = None):
        super().__init__(rate, dist, products)
        
    def __str__(self):
        return "GenReaction(rate={}, products={})".format(self.rate, self.products)
        
class UniReaction(Reaction):
    """ The UniReaction class represents unimolecular reactions. 
        
        Parameters:
        - spec: int
            The species undergoing the reaction
    """

    def __init__(self, rate, spec, products = None):
        super().__init__(rate, products)
        self.spec = spec
        
    def __str__(self):
        return "UniReaction(rate={}, spec={}, products={})".format(self.rate, self.spec, self.products)
        
class BiReaction(Reaction):
    """
        The BiReaction class represents bimolecular reactions.

        Parameters:
        - specA, specB: int
            The species of the two particles undergoing the reaction
    """

    def __init__(self, rate, specA, specB, products = None):
        super().__init__(rate, products)
        self.specA = specA
        self.specB = specB
        
    def __str__(self):
        return "BiReaction(rate={}, specA={}, specB={}, products={})".format(self.rate, self.specA, self.specB, self.products)
    
class ReactionSystem:
    """ This class stores information about a reaction system.
    
        Arguments:
            n_species: int
                Number of species in the system. Species are 
                enumerated starting from 0.
               
            reactions: array of reactions (default: ())
                List of allowed reactions in the system
            
            initial_state: array of ints
                Initial number of particles per species. 
    """
    def __init__(self, n_species, reactions = (), initial_state = None):
        self.n_species = n_species
        
        if initial_state is None:
            initial_state = np.zeros(n_species, dtype=int)
        
        self.initial_state = initial_state
        
        self.reactions = reactions
        self.sort_reactions(reactions)
        
    def sort_reactions(self, reactions):
        """ Sort reactions by type during initialisation """
        self.reactions_gen = []
        self.reactions_uni = []
        self.reactions_bi = []
        
        for reac in reactions:
            if isinstance(reac, GenReaction):
                self.reactions_gen.append(reac)
            elif isinstance(reac, UniReaction):
                self.reactions_uni.append(reac)
            elif isinstance(reac, BiReaction):
                self.reactions_bi.append(reac)
            else:
                raise TypeError("Unknown reaction type for '{}'".format(reac))
        
    def create_particle_system(self, seed=None):
        """ Create a new instance of ParticleSystem associated to this reaction system """
        return ParticleSystem(self, seed=seed)
    
class ParticleSystem:
    """ This class represents the state of a particle system associated to a ReactionSystem 
    
        Args:
            system: ReactionSystem
                Associated reaction system defining the model to use
                
            seed: int or None (default: None)
                Optional seed for random number generator
                
        Attributes:
            counts: n_species array of ints
                Per-species particle counts
                
            t: float
                Current time in the system
                
            events: array of events
                Ordered list of time-stamped events in the system
    """
    def __init__(self, system, seed=None):
        self.system = system
        
        self.counts = np.zeros(system.n_species, dtype=int)
        
        self.rng = np.random.RandomState(seed=seed)
       
        # Precompute some things for ease of access during simulations
        self.gen_rates = np.asarray([ r.rate for r in self.system.reactions_gen ])
        
        self.uni_rates = np.asarray([ r.rate for r in self.system.reactions_uni ])
        self.uni_spec = np.asarray([ r.spec for r in self.system.reactions_uni ], dtype=int)
       
        self.t = 0
        self.events = []
        
        self.add_initial_molecules()
        
    ### INTERNALS ###
    def add_products(self, products):
        """ Add list of particles to the reaction system. Returns list of particles placed.
        """

        # Create list of products to be placed
        products = self.expand_products(products)
        
        log = []
        for product in products:
            self.counts[product] += 1
            
            log.append((product,))
            
        return log
    
    def expand_products(self, products):
        """ 
            Convert description of reaction products into list of products to be
            added, drawing particle numbers for products produced in bursts.
        """
        ret = []
        for prod in products:
            if type(prod) == int:
                ret.append(prod)
            else:
                b, spec = prod
                
                p = 1 / b
                
                # Numpy's geometric distribution starts at 1
                n = self.rng.geometric(p) - 1
                ret += [ spec for i in range(n) ]
                
        return ret
    
    def add_initial_molecules(self):
        """ Place initial molecules in the system """
        assert self.t == 0
        
        for spec, n_init in enumerate(self.system.initial_state):
            product_log = self.add_products([ spec for i in range(n_init) ])
                
            event = ("gen", None, product_log)
            self.events.append((0, event))
            
    def compute_bi_rates(self):
        """ Compute reaction rates for bimolecular reactions """
        rates = np.empty(len(self.system.reactions_bi),)
        
        for i, reac in enumerate(self.system.reactions_bi):
            # Do not overcount reactant pairs if both educts are of the same species
            if reac.specA == reac.specB:
                combs = 0.5 * self.counts[reac.specA] * (self.counts[reac.specA] - 1)
            else:
                combs = self.counts[reac.specA] * self.counts[reac.specB]
                
            rates[i] = reac.rate * combs
            
        return rates
    
    ### UPDATES ###
    def perform_gen_reaction(self, reaction):
        """ Simulate one occurrence of the specified generative reaction """
        product_log = self.add_products(reaction.products)
        return ("gen", reaction, product_log)
                        
    def perform_uni_reaction(self, reaction):
        """ Simulate one occurrence of the specified unimolecular reaction """
        self.counts[reaction.spec] -= 1
        product_log = ()
        
        if reaction.products is not None:
            product_log = self.add_products(reaction.products)
            
        return ("uni", reaction, product_log)
    
    def perform_bi_reaction(self, reaction, rates):
        """ Simulate one occurrence of the specified bimolecular reaction.
            Refer to perform_uni_reaction for more information. """
        
        self.counts[reaction.specA] -= 1
        self.counts[reaction.specB] -= 1
        
        product_log = ()
        if reaction.products is not None:
            product_log = self.add_products(reaction.products)
            
        return ("bi", reaction, product_log)
        
    ### MAIN LOOP ###
    def run(self, tmax, disable_pbar=True):
        """ Run simulation for tmax time units using the Gillespie algorithm """
        #if kwargs["extras"] is not None:
        #    params = kwargs["extras"]

        t0 = self.t
        
        gen_rates = self.gen_rates
        
        with tqdm(total=tmax, 
                  desc="Time simulated: ", 
                  unit="s", 
                  disable=disable_pbar) as pbar:
            while True:
                uni_rates = self.uni_rates * self.counts[self.uni_spec]
                bi_rates = self.compute_bi_rates()
                rate = np.sum(gen_rates) + np.sum(uni_rates) + np.sum(bi_rates)
                
                # Nothing happening
                if rate == 0.0 or not np.isfinite(rate):
                    if not np.isfinite(rate):
                        logger.warning("Numerical blow-up in CME simulation")
                       
                    # Pretend the last reaction happened at time tmax.
                    # This is necessary for updating progress bar correctly.
                    dt = 0
                    break

                dt = self.rng.exponential(1 / rate)
                self.t += dt
                if self.t > t0 + tmax:
                    break
                    
                pbar.update(dt)

                # The Gillespie algorithm randomly samples a possible event
                # with probabilities proportional to the rates
                p = self.rng.uniform(0, rate)

                # Zero-molecular reaction happening
                if p <= np.sum(gen_rates):
                    for reac, rate in zip(self.system.reactions_gen, gen_rates):
                        if p >= rate:
                            p -= rate
                            continue

                        event = self.perform_gen_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Unimolecular reaction happening
                elif p <= np.sum(gen_rates) + np.sum(uni_rates): 
                    p -= np.sum(gen_rates)
                    
                    for reac in self.system.reactions_uni:
                        if p >= reac.rate * self.counts[reac.spec]:
                            p -= reac.rate * self.counts[reac.spec]
                            continue

                        event = self.perform_uni_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Bimolecular reaction happening
                else:
                    p -= np.sum(gen_rates) + np.sum(uni_rates)
                    
                    for reac, rates in zip(self.system.reactions_bi, bi_rates):
                        if p >= np.sum(rates):
                            p -= np.sum(rates)
                            continue
                            
                        event = self.perform_bi_reaction(reac, rates)
                        self.events.append((self.t, event))
                        break
           
            # This set sprogress bar value to t0 + tmax
            pbar.update(dt - (self.t - t0 - tmax))
            self.t = t0 + tmax

    ### MAIN LOOP ###
    def run_time(self, tmax, disable_pbar=True, **kwargs):
        """ Run simulation for tmax time units using the Gillespie algorithm """
        t0 = self.t
        # Applicabale to kb=kb0/(t+tau)**h as the only bimolecular reaction
        assert len(self.system.reactions_bi) == 1
        params = kwargs['extra']
        assert len(params) == 3
        kb0 = params[0]
        tau = params[1]
        h = params[2]
        # print("Time", t0)

        gen_rates = self.gen_rates

        with tqdm(total=tmax,
                  desc="Time simulated: ",
                  unit="s",
                  disable=disable_pbar) as pbar:
            while True:
                uni_rates = self.uni_rates * self.counts[self.uni_spec]
                # print("Uni rates", uni_rates)
                # Valid for only 1 bimolecular reaction with rate 1
                bi_rates = self.compute_bi_rates()*kb0/(self.t+tau)**h
                # print("Bi rates", bi_rates)
                rate = np.sum(gen_rates) + np.sum(uni_rates) + np.sum(bi_rates)
                B = rate
                # print("a0 ", rate)
                # Because the bimolecular propensity function is monotonically decreasing we set B=rate
                L = tmax + t0 - self.t

                # Nothing happening
                if rate == 0.0 or not np.isfinite(rate):
                    if not np.isfinite(rate):
                        logger.warning("Numerical blow-up in CME simulation")

                    # Pretend the last reaction happened at time tmax.
                    # This is necessary for updating progress bar correctly.
                    dt = 0
                    break
                dt = self.rng.exponential(1 / rate)
                # print("Tau", dt)
                if dt > L:
                    self.t += L
                else:
                    self.t += dt
                # print("t0 ", t0)
                # print("time", self.t)
                # print("end", tmax)
                if self.t >= t0 + tmax:
                    # print("Time for a break!")
                    break

                if dt < L:
                    # The Gillespie algorithm randomly samples a possible event
                    # with probabilities proportional to the rates
                    # p = self.rng.uniform(0, rate)
                    p = self.rng.uniform(0, B)

                    # The 'normal' reaction fires
                    if rate >= p:
                        # print("Reaction happens!")
                        # Zero-molecular reaction happening
                        if p <= np.sum(gen_rates):
                            for reac, rate in zip(self.system.reactions_gen, gen_rates):
                                if p >= rate:
                                    p -= rate
                                    continue

                                event = self.perform_gen_reaction(reac)
                                self.events.append((self.t, event))
                                break

                        # Unimolecular reaction happening
                        elif p <= np.sum(gen_rates) + np.sum(uni_rates):
                            p -= np.sum(gen_rates)

                            for reac in self.system.reactions_uni:
                                if p >= reac.rate * self.counts[reac.spec]:
                                    p -= reac.rate * self.counts[reac.spec]
                                    # print("Unimolecular reaction")
                                    continue

                                event = self.perform_uni_reaction(reac)
                                self.events.append((self.t, event))
                                break

                        # Bimolecular reaction happening (valid for only 1 bimolecular reaction)
                        else:
                            p -= np.sum(gen_rates) + np.sum(uni_rates)

                            event = self.perform_bi_reaction(self.system.reactions_bi[0], bi_rates)
                            self.events.append((self.t, event))

            pbar.update(dt)
            # This set sprogress bar value to t0 + tmax
            pbar.update(dt - (self.t - t0 - tmax))
            self.t = t0 + tmax

    ### MAIN LOOP ###
    def run_time_3(self, tmax, disable_pbar=True, **kwargs):
        """ Run simulation for tmax time units using the Gillespie algorithm """
        t0 = self.t
        # Applicabale to kb=kb0/(t+tau)**h as 3 first bimolecular reactions
        assert len(self.system.reactions_bi) == 3
        params = kwargs['extra']
        assert len(params) == 9
        k10 = params[0]
        tau1 = params[1]
        h1 = params[2]
        k20 = params[3]
        tau2 = params[4]
        h2 = params[5]
        k30 = params[6]
        tau3 = params[7]
        h3 = params[8]
        # print("Time", t0)

        gen_rates = self.gen_rates

        with tqdm(total=tmax,
                  desc="Time simulated: ",
                  unit="s",
                  disable=disable_pbar) as pbar:
            while True:
                uni_rates = self.uni_rates * self.counts[self.uni_spec]
                # print("Uni rates", uni_rates)
                # Valid for only 3 bimolecular reactions with rates 1
                bi_rates = self.compute_bi_rates() * np.array([k10/(self.t+tau1)**h1, k20/(self.t+tau2)**h2, k30/(self.t+tau3)**h3])
                # print("Bi rates", bi_rates)
                rate = np.sum(gen_rates) + np.sum(uni_rates) + np.sum(bi_rates)
                B = rate
                # print("a0 ", rate)
                # Because the bimolecular propensity function is monotonically decreasing we set B=rate
                L = tmax + t0 - self.t

                # Nothing happening
                if rate == 0.0 or not np.isfinite(rate):
                    if not np.isfinite(rate):
                        logger.warning("Numerical blow-up in CME simulation")

                    # Pretend the last reaction happened at time tmax.
                    # This is necessary for updating progress bar correctly.
                    dt = 0
                    break
                dt = self.rng.exponential(1 / rate)
                # print("Tau", dt)
                if dt > L:
                    self.t += L
                else:
                    self.t += dt
                # print("t0 ", t0)
                # print("time", self.t)
                # print("end", tmax)
                if self.t >= t0 + tmax:
                    # print("Time for a break!")
                    break

                if dt < L:
                    # The Gillespie algorithm randomly samples a possible event
                    # with probabilities proportional to the rates
                    # p = self.rng.uniform(0, rate)
                    p = self.rng.uniform(0, B)

                    # The 'normal' reaction fires
                    if rate >= p:
                        # print("Reaction happens!")
                        # Zero-molecular reaction happening
                        if p <= np.sum(gen_rates):
                            for reac, rate in zip(self.system.reactions_gen, gen_rates):
                                if p >= rate:
                                    p -= rate
                                    continue

                                event = self.perform_gen_reaction(reac)
                                self.events.append((self.t, event))
                                break

                        # Unimolecular reaction happening
                        elif p <= np.sum(gen_rates) + np.sum(uni_rates):
                            p -= np.sum(gen_rates)

                            for reac in self.system.reactions_uni:
                                if p >= reac.rate * self.counts[reac.spec]:
                                    p -= reac.rate * self.counts[reac.spec]
                                    # print("Unimolecular reaction")
                                    continue

                                event = self.perform_uni_reaction(reac)
                                self.events.append((self.t, event))
                                break

                        # Bimolecular reaction happening (valid for only for bimolecular reactions initialised in the beginning)
                        else:
                            p -= np.sum(gen_rates) + np.sum(uni_rates)

                            for reac, rates in zip(self.system.reactions_bi, bi_rates):
                                if p >= np.sum(rates):
                                    p -= np.sum(rates)
                                    continue

                                event = self.perform_bi_reaction(reac, rates)
                                self.events.append((self.t, event))
                                break
            pbar.update(dt)
            # This set sprogress bar value to t0 + tmax
            pbar.update(dt - (self.t - t0 - tmax))
            self.t = t0 + tmax

    ### MAIN LOOP ###
    def run_time_solve(self, tmax, disable_pbar=True, **kwargs):
        """ Run simulation for tmax time units using the modified Gillespie algorithm """
        t0 = self.t
        params = kwargs['extra']
        assert len(params) == 3
        kb0 = params[0]
        tau_p = params[1]
        h = params[2]

        gen_rates = self.gen_rates
        with tqdm(total=tmax,
                  desc="Time simulated: ",
                  unit="s",
                  disable=disable_pbar) as pbar:
            while True:
                uni_rates = self.uni_rates * self.counts[self.uni_spec]
                bi_rates = self.compute_bi_rates()

                rgen = np.log(1 / self.rng.uniform())
                func = lambda tau: (np.sum(gen_rates) + np.sum(uni_rates)) * tau - np.log(1/rgen) + (1/(-1 + h))*kb0*((t0 + tau_p)*(t0 + tau + tau_p))**(-h)*(-(t0 + tau_p)**h*(t0 + tau + tau_p) + (t0 + tau_p)*(t0 + tau + tau_p)**h)
                dt = fsolve(func, 1)[0]
                print(dt)
                # dt = r
                multiplier = kb0/(t0 + dt + tau_p)**h
                rate = np.sum(gen_rates) + np.sum(uni_rates) + np.sum(bi_rates) * multiplier
                # Nothing happening
                if rate == 0.0 or not np.isfinite(rate):
                    if not np.isfinite(rate):
                        logger.warning("Numerical blow-up in CME simulation")

                    # Pretend the last reaction happened at time tmax.
                    # This is necessary for updating progress bar correctly.
                    dt = 0
                    break
                self.t += dt
                if self.t > t0 + tmax:
                    break

                pbar.update(dt)

                # The Gillespie algorithm randomly samples a possible event
                # with probabilities proportional to the rates
                p = self.rng.uniform(0, rate)

                # Zero-molecular reaction happening
                if p <= np.sum(gen_rates):
                    for reac, rate in zip(self.system.reactions_gen, gen_rates):
                        if p >= rate:
                            p -= rate
                            continue

                        event = self.perform_gen_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Unimolecular reaction happening
                elif p <= np.sum(gen_rates) + np.sum(uni_rates):
                    p -= np.sum(gen_rates)

                    for reac in self.system.reactions_uni:
                        if p >= reac.rate * self.counts[reac.spec]:
                            p -= reac.rate * self.counts[reac.spec]
                            continue

                        event = self.perform_uni_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Bimolecular reaction happening
                else:
                    p -= np.sum(gen_rates) + np.sum(uni_rates)

                    for reac, rates in zip(self.system.reactions_bi, bi_rates):
                        if p >= np.sum(rates*multiplier):
                            p -= np.sum(rates*multiplier)
                            continue

                        event = self.perform_bi_reaction(reac, rates)
                        self.events.append((self.t, event))
                        break

            # This set sprogress bar value to t0 + tmax
            pbar.update(dt - (self.t - t0 - tmax))
            self.t = t0 + tmax
    
    def get_dist(self, t_max=None):
        """ 
            Return the distribution of particle numbers over the lifetime of the system
            using time-averaging.
        """
        counts = np.zeros((self.system.n_species, len(self.events) + 1), dtype=int)
        weights = np.zeros(len(self.events) + 1)

        if t_max is None:
            t_max = self.t
            
        t_last = 0
        i = 0
        
        for t, e in self.events:
            if t > t_max:
                continue

            weights[i] = t - t_last
            t_last = t

            i += 1
            counts[:,i] = counts[:,i-1]
            if e[0] == "gen":
                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1
            if e[0] == "uni":
                reac = e[1]
                counts[reac.spec][i] -= 1

                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1
            elif e[0] == "bi":
                reac = e[1]

                counts[reac.specA][i] -= 1
                counts[reac.specB][i] -= 1

                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1

        weights[i] = t_max - t_last

        return pdist.ParticleDistribution(counts[:,:i+1], weights=weights[:i+1])

    def get_dist_samples(self, event_list, t_max=None, n_samples=1, max_iter=1, pairs=False):
        """
            Return the distribution of particle numbers over the lifetime of the system
            for multiple samples.
        """
        # Calculate indexes for each iteration separately
        index = np.zeros(max_iter, dtype=int)
        # Calculate maximal size of the array (including the ends)
        event_num = sum([len(listElem) + 1 for listElem in event_list])
        counts = np.zeros((self.system.n_species, event_num, max_iter), dtype=int)
        weights = np.zeros((event_num, max_iter))
        for sample, events in enumerate(event_list):
            for iter in range(max_iter):
                i = index[iter]
                t_last = 0
                for t, e in events:
                    if t > t_max*(iter + 1):
                        continue

                    weights[i, iter] = t - t_last
                    t_last = t

                    i += 1
                    counts[:, i, iter] = counts[:, i - 1, iter]
                    if e[0] == "gen":
                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1
                    if e[0] == "uni":
                        reac = e[1]
                        counts[reac.spec][i][iter] -= 1

                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1
                    elif e[0] == "bi":
                        reac = e[1]

                        counts[reac.specA][i][iter] -= 1
                        counts[reac.specB][i][iter] -= 1

                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1

                weights[i][iter] = t_max*(iter + 1) - t_last
                # Account for the end of the iteration
                index[iter] = i + 1
        dist_list = []
        for iter in range(max_iter):
            dist_list.append(pdist.ParticleDistribution(counts[:, :index[iter], iter], weights=weights[:index[iter], iter]))
        return dist_list


    def get_dist_samples_separate(self, event_list, t_max=None, n_samples=1, max_iter=1, pairs=False):
        """
            Return the distribution of particle numbers over the lifetime of the system
            for multiple samples.
        """
        # Calculate indexes for each iteration separately
        index = np.zeros(max_iter, dtype=int)
        # Calculate maximal size of the array (including the ends)
        event_num = sum([len(listElem) + 1 for listElem in event_list])
        counts = np.zeros((self.system.n_species, event_num, max_iter), dtype=int)
        weights = np.zeros((event_num, max_iter))
        for sample, events in enumerate(event_list):
            for iter in range(max_iter):
                i = index[iter]
                t_last = 0
                for t, e in events:
                    if t > t_max*(iter + 1):
                        continue

                    weights[i, iter] = t - t_last
                    t_last = t

                    i += 1
                    counts[:, i, iter] = counts[:, i - 1, iter]
                    if e[0] == "gen":
                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1
                    if e[0] == "uni":
                        reac = e[1]
                        counts[reac.spec][i][iter] -= 1

                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1
                    elif e[0] == "bi":
                        reac = e[1]

                        counts[reac.specA][i][iter] -= 1
                        counts[reac.specB][i][iter] -= 1

                        product_log = e[2]
                        for spec_product in product_log:
                            counts[spec_product][i][iter] += 1

                weights[i][iter] = t_max*(iter + 1) - t_last
                # Account for the end of the iteration
                index[iter] = i + 1
        dist_list = []
        for iter in range(max_iter):
            for part_num in range(np.shape(counts)[0]):
                dist_list.append(pdist.ParticleDistribution(counts[part_num, :index[iter], iter], weights=weights[:index[iter], iter]))
        return dist_list
