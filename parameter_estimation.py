import random
from numpy.random import choice
from sklearn import linear_model
import numpy as np
import scipy.stats as st
import agent
from random import sample

from copy import deepcopy
from copy import copy
import logging
import sys
import train_data
import matplotlib.pyplot as plt

logging.basicConfig(filename='parameter_estimation.log', format='%(asctime)s %(message)s', level=logging.DEBUG)


radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0

types = ['l1', 'l2']#, 'f1', 'f2', 'w']

########################################################################################################################
class Parameter:
    def __init__(self, level, angle, radius):
        self.level = level
        self.angle = angle
        self.radius = radius
        self.iteration = 0
        self.min_max = [(0, 1), (0.1, 1), (0.1, 1)]

    def update(self, added_value):
        self.level += added_value[0]
        self.angle += added_value[1]
        self.radius += added_value[2]
        return self


########################################################################################################################
class TypeEstimation:
    def __init__(self, a_type, generated_data_number, PF_add_threshold, train_mode, mutation_rate,unknown_agent, sim):
        self.type = a_type  # Type for which we are doing estimation
        self.type_probability = 0
        self.type_probabilities = []
        self.estimation_history = []
        self.action_probabilities = []
        self.internal_state = None
        self.train_mode = train_mode
        self.train_data = train_data.TrainData(generated_data_number, PF_add_threshold, train_mode, a_type,
                                               mutation_rate, unknown_agent, sim)

    ####################################################################################################################
    def add_estimation_history(self, probability, level, angle, radius):
        new_parameter = Parameter(level, angle, radius)
        self.estimation_history.append(new_parameter)
        self.type_probabilities.append(probability)

    ####################################################################################################################

    def get_estimation_history(self):
        estimation_historty = "["
        for est_hist in self.estimation_history:
            estimation_historty += "[" + str(est_hist.level) + "," + str(est_hist.angle) + "," + str(
                est_hist.radius) + "],"

        estimation_historty = estimation_historty[0:len(estimation_historty) - 1]
        estimation_historty += "]"
        return estimation_historty

    ####################################################################################################################
    def get_last_type_probability(self):
        if len(self.type_probabilities) > 0:
            return self.type_probabilities[-1]
        else:
            return 1/len(types)

    ####################################################################################################################
    def get_last_estimation(self):
        if len(self.estimation_history) > 0:
            return self.estimation_history[-1]
        else:
            return sample(self.train_data.data_set,1)[0]
    
    ####################################################################################################################
    def get_last_action_probability(self):
        if len(self.action_probabilities) > 0:
            return self.action_probabilities[-1]
        else:
            return 0.2

    ####################################################################################################################
    def update_estimation(self, estimation, action_probability):
        self.estimation_history.append(estimation)
        self.action_probabilities.append(action_probability)


########################################################################################################################
class ParameterEstimation:

    def __init__(self,  generated_data_number, PF_add_threshold, train_mode, apply_adversary,
                  mutation_rate, uknown_agent, sim):

        # P(teta|H)
        self.apply_adversary = apply_adversary
        if self.apply_adversary:
            self.w_estimation = TypeEstimation('w', generated_data_number, PF_add_threshold, train_mode, mutation_rate, uknown_agent, sim)

        self.l1_estimation = TypeEstimation('l1',  generated_data_number, PF_add_threshold, train_mode, mutation_rate, uknown_agent, sim)
        self.l2_estimation = TypeEstimation('l2',  generated_data_number, PF_add_threshold, train_mode, mutation_rate, uknown_agent, sim)
        self.f1_estimation = TypeEstimation('f1',  generated_data_number, PF_add_threshold, train_mode, mutation_rate, uknown_agent, sim)
        self.f2_estimation = TypeEstimation('f2',  generated_data_number, PF_add_threshold, train_mode, mutation_rate, uknown_agent, sim)


        self.action_history = []
        self.train_mode = train_mode
        self.actions_to_reach_target = []
        # type_selection_mode are: all types selection 'AS', Posterior Selection 'PS' , Bandit Selection 'BS'
        self.type_selection_mode = None

        # Parameter estimation mode is AGA if it is Approximate Gradient Ascent ,
        #                              ABU if it is Approximate Bayesian Updating
        self.parameter_estimation_mode = None
        self.polynomial_degree = None

        self.iteration = 0
        self.belief_poly = [None] * 3

    ####################################################################################################################
    # Initialisation random values for parameters of each type and probability of actions in time step 0

    def estimation_configuration(self, type_selection_mode, parameter_estimation_mode, polynomial_degree, type_estimation_mode):

        self.type_selection_mode = type_selection_mode
        self.parameter_estimation_mode = parameter_estimation_mode
        self.type_estimation_mode = type_estimation_mode
        self.polynomial_degree = polynomial_degree

    ####################################################################################################################
    # Initialisation random values for parameters of each type and probability of actions in time step 0
    def estimation_initialisation(self):
        if self.apply_adversary:
            l1_init_prob = 0.2
            l2_init_prob = 0.2
            f1_init_prob = 0.2
            f2_init_prob = 0.2
            w_init_prob = 0.2

        else:
            l1_init_prob = 0.25
            l2_init_prob = 0.25
            f1_init_prob = 0.25
            f2_init_prob = 0.25

        self.l1_estimation.add_estimation_history(round(l1_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.l2_estimation.add_estimation_history(round(l2_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.f1_estimation.add_estimation_history(round(f1_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.f2_estimation.add_estimation_history(round(f2_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        if self.apply_adversary:
            self.w_estimation.add_estimation_history(round(w_init_prob, 2),
                                              round(random.uniform(level_min, level_max), 2),  # 'level'
                                              round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                              round(random.uniform(angle_min, angle_max), 2))  # 'angle'

    ####################################################################################################################
    def get_sampled_probability(self):

        type_probes = list()
        type_probes.append(self.l1_estimation.get_last_type_probability())
        type_probes.append(self.l2_estimation.get_last_type_probability())
        type_probes.append(self.f1_estimation.get_last_type_probability())
        type_probes.append(self.f2_estimation.get_last_type_probability())
        if self.apply_adversary:
            type_probes.append(self.w_estimation.get_last_type_probability())

        selected_type = choice(types, p=type_probes)  # random sampling the action

        return selected_type

    ####################################################################################################################
    def get_highest_type_probability(self):

        highest_probability = -1
        selected_type = ''

        for type in types:
            if type == 'l1':
                tmp_prob = self.l1_estimation.get_last_type_probability()

            if type == 'l2':
                tmp_prob = self.l2_estimation.get_last_type_probability()

            if type == 'f1':
                tmp_prob = self.f1_estimation.get_last_type_probability()

            if type == 'f2':
                tmp_prob = self.f2_estimation.get_last_type_probability()

            if self.apply_adversary:
                if type == 'w':
                    tmp_prob = self.w_estimation.get_last_type_probability()

            if tmp_prob > highest_probability:
                highest_probability = tmp_prob
                selected_type = type

        return selected_type

    ####################################################################################################################
    def get_parameters_for_selected_type(self, selected_type):

        if selected_type == 'l1':
            return self.l1_estimation.get_last_estimation()

        if selected_type == 'l2':
            return self.l2_estimation.get_last_estimation()

        if selected_type == 'f1':
            return self.f1_estimation.get_last_estimation()

        if selected_type == 'f2':
            return self.f2_estimation.get_last_estimation()

        if selected_type == 'w':
            return self.w_estimation.get_last_estimation()

    #################################################################################################################
    def get_parameter(self, parameter, index):
        #TODO: Level = 0, angle = 1, radius = 2? Perhaps there should be a nicer way to do this

        if index == 0:
            return parameter.level
        if index == 1:
            return parameter.angle
        if index == 2:
            return parameter.radius

    ####################################################################################################################
    def calculate_EGO(self, agent_type, time_step):  # Exact Global Optimisation

        multiple_results = 1
        if agent_type.agent_type == 'l1':
            for i in range(0,time_step):
                multiple_results = multiple_results #* self.p_action_parameter_type_l1[i]

        if agent_type.agent_type == 'l2':
            self.p_action_parameter_type_l2 = []

        if agent_type.agent_type == 'f1':
            self.p_action_parameter_type_f1 = []

        if agent_type.agent_type == 'f2':
            self.p_action_parameter_type_f2 = []

        if agent_type.agent_type == 'w':
            self.p_action_parameter_type_w = []

        return

    ####################################################################################################################
    def multivariate_bayesian(self, x_train, y_train, previous_estimate):
        # TODO: This method must be called once, not four times in a loop as is currently the case
        np.random.seed(123)

        # Fit multivariate polynomial of degree 4PF_O_2
        f_poly = linear_model.LinearRegression(fit_intercept=True)
        f_poly.fit(x_train, y_train)

        # Extract polynomial coefficients
        f_coefficients = np.insert(f_poly.coef_, 0, f_poly.intercept_)
        logging.info('f-hat Coefficients: {}'.format(f_coefficients))

        # Generate prior
        if self.iteration == 0:
            beliefs = st.uniform.rvs(0, 1, size=4)
            logging.info('Randomly Sampling Beliefs From Standard Uniform')
        else:
            beliefs = previous_estimate.observation_history[-1]

        logging.info('Beliefs at Iteration {}: {}'.format(previous_estimate.iteration, beliefs))

        # Catch array broadcasting errors
        assert len(beliefs) == len(f_coefficients), 'F coefficient and beliefs of differing lengths'
        if len(beliefs) != len(f_coefficients):
            logging.warning('Iteration {}, beliefs and f-hat coefficients of differing lengths.'.format(self.iteration))
            logging.warning('Beliefs Length: {}\nCoefficients Length: {}'.format(len(beliefs), len(f_coefficients)))

        # Compute Improper Posterior Posterior
        g_hat = f_coefficients * beliefs
        logging.info('Polynomial Convolution g-hat values: {}'.format(g_hat))

        # Collect samples from g
        sampled_x = np.linspace(0, 1, 4)
        sampled_y = st.uniform.rvs(0.1, 1, 4)  # TODO: How can I get g(p^l) here?

        # Fit h-hat
        h_polynomial = linear_model.LinearRegression(fit_intercept=True)
        h_polynomial.fit(sampled_x, sampled_y)
        h_coefficients = np.insert(h_polynomial.coef_, 0, h_polynomial.intercept_)

        # Integrate h-hat
        def integrand(level, radius, angle, x):
            pass

        logging.info('Estimation Complete\n{}'.format('-' * 100))

    ####################################################################################################################
    def findMin(self,polynomial):
        derivative = polynomial.deriv()

        roots = derivative.roots()

        minValue = sys.maxsize

        for r in roots:
            if polynomial(r) < minValue:
                minValue = polynomial(r)

        if polynomial(polynomial.domain[0]) < minValue:
            minValue = polynomial(polynomial.domain[0])

        if polynomial(polynomial.domain[1]) < minValue:
            minValue = polynomial(polynomial.domain[1])

        return minValue

    ####################################################################################################################
    def inversePolynomial(self,polynomialInput, y):
        solutions = list()

        polynomial = polynomialInput.copy()
        
        polynomial.coef[0] = polynomial.coef[0] - y

        roots = polynomial.roots()

        for r in roots:
            if (r >= polynomial.domain[0] and r <= polynomial.domain[1]):
                if (not (isinstance(r,complex))):
                    solutions.append(r)
                elif (r.imag == 0):
                    solutions.append(r.real)

        ## We should always have one solution for the inverse?
        if (len(solutions) > 1):
            print "Warning! Multiple solutions when sampling for ABU"
        
        return solutions[0]

    ####################################################################################################################
    # Inverse transform sampling
    # https://en.wikipedia.org/wiki/Inverse_transform_sampling
    def sampleFromBelief(self,polynomial,sizeList):
        returnMe = [None]*sizeList

        ## To calculate the CDF, I will first get the integral. The lower part is the lowest possible value for the domain
        ## Given a value x, the CDF will be the integral at x, minus the integral at the lowest possible value.
        dist_integ = polynomial.integ()
        lower_part = dist_integ(polynomial.domain[0])
        cdf = dist_integ.copy()
        cdf.coef[0] = cdf.coef[0] - lower_part
    
        for s in range(sizeList):
            u = np.random.uniform(0, 1)

            returnMe[s] = self.inversePolynomial(cdf, u)

        return returnMe

    ####################################################################################################################
    def update_action_history(self, current_action):

        self.action_history.append(current_action)
        if current_action == 'L':
            self.actions_to_reach_target = []
        else:
            self.actions_to_reach_target.append(current_action)

    ####################################################################################################################
    def nested_list_sum(self, nested_lists):
        if type(nested_lists) == list:
            return np.sum(self.nested_list_sum(sublist) for sublist in nested_lists)
        else:
            return 1

    ####################################################################################################################
    def UCB_selection(self, time_step, final=False):
        if self.apply_adversary:
            agent_types = ['l1', 'l2', 'f1', 'f2', 'w']
        else:
            agent_types = ['l1', 'l2', 'f1', 'f2']

        # Get the total number of probabilities
        prob_count = self.nested_list_sum(agent_types)

        # Return the mean probability for each type of bandit
        mean_probabilities = [np.mean(i) for i in agent_types]

        # Confidence intervals from standard UCB formula
        cis = [np.sqrt((2 * np.log(prob_count)) / len(agent_types[i]) + 0.01) for i in range(len(agent_types))]

        # Sum together means and CIs
        ucb_values = np.array(mean_probabilities) + np.array(cis)

        # Get max UCB value
        max_index = np.argmax(ucb_values)

        # Determine Agent Type to return
        try:
            if max_index == 0:
                return_agent = ['l1']
            elif max_index == 1:
                return_agent = ['l2']
            elif max_index == 2:
                return_agent = ['f1']
            elif max_index == 3:
                return_agent = ['f2']
            elif max_index == 4:
                return_agent = ['w']

            else:
                print('UCB has not worked correctly, defaulting to l1')
                return_agent = ['l1']
        except:
            print('An error has occured in UCB, resorting to l1')
            return_agent = ['f1']

        #print('UCB Algorithm returned agent of type: {}'.format(return_agent[0]))

        if final:
            return return_agent
        else:
            return ['f2']

    ####################################################################################################################
    def set_choose_target_state(self, state, agent_type):
        if agent_type == 'l1':
            self.l1_estimation.choose_target_state = state
        elif agent_type == 'l2':
            self.l2_estimation.choose_target_state = state
        elif agent_type == 'f1':
            self.f1_estimation.choose_target_state = state
        elif agent_type == 'f2':
            self.f2_estimation.choose_target_state = state
        elif agent_type == 'w':
            self.w_estimation.choose_target_state = state

    ####################################################################################################################
    def normalize_type_probability(self,type):
        # 1. Defining the values
        # print 'Normalizing:',self.l1_estimation.type_probability , self.l2_estimation.type_probability,self.f1_estimation.type_probability, self.f2_estimation.type_probability

        l1_update_belief_value = self.l1_estimation.type_probability
        l2_update_belief_value = self.l2_estimation.type_probability
        f1_update_belief_value = self.f1_estimation.type_probability
        f2_update_belief_value = self.f2_estimation.type_probability
        if self.apply_adversary:
            w_update_belief_value = self.w_estimation.type_probability

        # 2. Summing
        sum_of_probabilities = l1_update_belief_value + l2_update_belief_value + \
                               f1_update_belief_value + f2_update_belief_value
        if self.apply_adversary :
            sum_of_probabilities += w_update_belief_value
        belief_factor = 1

        # 3. Normalising
        if sum_of_probabilities != 0:
            belief_factor = 1 / sum_of_probabilities

            w_prob = 0
            l1_prob = l1_update_belief_value * belief_factor
            l2_prob = l2_update_belief_value * belief_factor
            f1_prob = f1_update_belief_value * belief_factor
            f2_prob = f2_update_belief_value * belief_factor

            if self.apply_adversary:
                w_prob = w_update_belief_value * belief_factor

        else:
            if self.apply_adversary:

                l1_prob = 0.2
                l2_prob = 0.2
                f1_prob = 0.2
                f2_prob = 0.2
                w_prob = 0.2

            else:
                l1_prob = 0.25
                l2_prob = 0.25
                f1_prob = 0.25
                f2_prob = 0.25
        if type =='l1':
            return l1_prob
        elif type == 'l2':
            return  l2_prob
        elif type =='f1':
            return f1_prob
        elif type == 'f2':
            return  f2_prob

    ####################################################################################################################
    def normalize_type_probabilities(self):
        # 1. Defining the values
        # print 'Normalizing:',self.l1_estimation.type_probability , self.l2_estimation.type_probability,self.f1_estimation.type_probability, self.f2_estimation.type_probability

        l1_update_belief_value = self.l1_estimation.type_probability
        l2_update_belief_value = self.l2_estimation.type_probability
        f1_update_belief_value = self.f1_estimation.type_probability
        f2_update_belief_value = self.f2_estimation.type_probability
        if self.apply_adversary:
            w_update_belief_value = self.w_estimation.type_probability

        # 2. Summing
        sum_of_probabilities = l1_update_belief_value + l2_update_belief_value + \
                               f1_update_belief_value + f2_update_belief_value
        if self.apply_adversary :
            sum_of_probabilities += w_update_belief_value
        belief_factor = 1

        # 3. Normalising
        if sum_of_probabilities != 0:
            belief_factor = 1 / sum_of_probabilities

            w_prob = 0
            l1_prob = l1_update_belief_value * belief_factor
            l2_prob = l2_update_belief_value * belief_factor
            f1_prob = f1_update_belief_value * belief_factor
            f2_prob = f2_update_belief_value * belief_factor

            self.l1_estimation.type_probability = l1_prob
            self.l2_estimation.type_probability = l2_prob
            self.f1_estimation.type_probability = f1_prob
            self.f2_estimation.type_probability = f2_prob

            self.l1_estimation.type_probabilities.append(l1_prob)
            self.l2_estimation.type_probabilities.append(l2_prob)
            self.f1_estimation.type_probabilities.append(f1_prob)
            self.f2_estimation.type_probabilities.append(f2_prob)

            if self.apply_adversary:
                w_prob = w_update_belief_value * belief_factor
                self.w_estimation.type_probability = w_prob
                self.w_estimation.type_probabilities.append(w_prob)
        else:
            if self.apply_adversary:

                self.l1_estimation.type_probability = 0.2
                self.l2_estimation.type_probability = 0.2
                self.f1_estimation.type_probability = 0.2
                self.f2_estimation.type_probability = 0.2
                self.w_estimation.type_probability = 0.2

                self.l1_estimation.type_probabilities.append(0.2)
                self.l2_estimation.type_probabilities.append(0.2)
                self.f1_estimation.type_probabilities.append(0.2)
                self.f2_estimation.type_probabilities.append(0.2)
                self.w_estimation.type_probabilities.append(0.2)
            else:

                self.l1_estimation.type_probability = 0.25
                self.l2_estimation.type_probability = 0.25
                self.f1_estimation.type_probability = 0.25
                self.f2_estimation.type_probability = 0.25

                self.l1_estimation.type_probabilities.append(0.25)
                self.l2_estimation.type_probabilities.append(0.25)
                self.f1_estimation.type_probabilities.append(0.25)
                self.f2_estimation.type_probabilities.append(0.25)

    ####################################################################################################################
    def bayesian_updating(self, x_train, y_train, previous_estimate,  polynomial_degree=2, sampling='average'):

        parameter_estimate = []

        for i in range(len(x_train[0])):
            # Get current independent variables
            current_parameter_set = [elem[i] for elem in x_train]

            # Obtain the parameter in questions upper and lower limits
            p_min = previous_estimate.min_max[i][0]
            p_max = previous_estimate.min_max[i][1]

            # Fit polynomial to the parameter being modelled
            f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y_train,
                                                              deg=polynomial_degree, full=False)
            
            f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly, domain=[p_min, p_max], window=[p_min, p_max])
            
            # Generate prior
            if self.iteration == 0:
                #beliefs = st.uniform.rvs(0, 6AGA_O_2, size=polynomial_degree + 6AGA_O_2)
                beliefs = [0]*(polynomial_degree + 1)
                beliefs[0] = 1.0/(p_max - p_min)
                
                current_belief_poly = np.polynomial.polynomial.Polynomial(coef=beliefs, domain=[p_min, p_max], window=[p_min,p_max])
            else:
                current_belief_poly = self.belief_poly[i]
            

            # Compute convolution
            g_poly = current_belief_poly*f_poly

            # Collect samples
            # Number of evenly spaced points to compute polynomial at
            # TODO: Not sure why it was polynomial_degree + 6AGA_O_2
            # spacing = polynomial_degree + 6AGA_O_2
            spacing = len(x_train)

            # Generate equally spaced points, unique to the parameter being modelled
            X = np.linspace(p_min, p_max, spacing)
            y = np.array([g_poly(j) for j in X])

            # Future polynomials are modelled using X and y, not D as it's simpler this way. I've left D in for now
            # TODO: possilby remove D if not needed at the end
            D = [(X[j], y[j]) for j in range(len(X))]

            # Fit h
            h_hat_coefficients = np.polynomial.polynomial.polyfit(X, y, deg=polynomial_degree, full=False)
            
            h_poly = np.polynomial.polynomial.Polynomial(coef=h_hat_coefficients, domain=[p_min, p_max], window=[p_min, p_max])

            # "Lift" the polynomial. Perhaps this technique is different than the one in Albrecht and Stone 2017.
            min_h = self.findMin(h_poly)
            if min_h < 0:
                h_poly.coef[0] = h_poly.coef[0] - min_h

            # Integrate h
            integration = h_poly.integ()

            # Compute I
            definite_integral = integration(p_max) - integration(p_min)

            # Update beliefs
            new_belief_coef = np.divide(h_poly.coef, definite_integral)  # returns an array
            new_belief = np.polynomial.polynomial.Polynomial(coef=new_belief_coef,domain=[p_min, p_max],window=[p_min, p_max])

            self.belief_poly[i] = new_belief

            # TODO: Not better to derivate and get the roots?
            if sampling == 'MAP':
                # Sample from beliefs
                polynomial_max = 0
                granularity = 1000
                x_vals = np.linspace(p_min, p_max, granularity)
                for j in range(len(x_vals)):
                    proposal = new_belief(x_vals[j])
                    #print('Proposal: {}'.format(proposal))
                    if proposal > polynomial_max:
                        polynomial_max = proposal

                parameter_estimate.append(polynomial_max)

            elif sampling == 'average':
                x_random = self.sampleFromBelief(new_belief, 10)
                parameter_estimate.append(np.mean(x_random))

            # Increment iterator

        new_parameter = Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
        #print('Parameter Estimate: {}'.format(parameter_estimate))
        self.iteration += 1

        return new_parameter

    ####################################################################################################################
    def calculate_gradient_ascent(self,x_train, y_train, old_parameter, polynomial_degree=2, univariate=True):
        # p is parameter estimation value at time step t-6AGA_O_2 and D is pair of (p,f(p))
        # f(p) is the probability of action which is taken by unknown agent with true parameters at time step t
        # (implementation of Algorithm 2 in the paper for updating parameter value)

        step_size = 0.05
        if not univariate:
            reg = linear_model.LinearRegression()
            reg.fit(x_train, y_train)
            gradient = reg.coef_

            # f_coefficients = np.polynomial.polynomial.polyfit(x_train, y_train,
            #                                                   deg=self.polynomial_degree, full=False)

            new_parameters = old_parameter.update(gradient * step_size)

            # Not sure if we need this rounding
            # new_parameters.level, new_parameters.angle, new_parameters.radius = \
            #    round(new_parameters.level, 2), round(new_parameters.angle, 2), round(new_parameters.radius, 2)

            if new_parameters.level < level_min:
                new_parameters.level = level_min

            if new_parameters.level > level_max:
                new_parameters.level = level_max

            if new_parameters.angle < angle_min:
                new_parameters.angle = angle_min

            if new_parameters.angle > angle_max:
                new_parameters.angle = angle_max

            if new_parameters.radius < radius_min:
                new_parameters.radius = radius_min

            if new_parameters.radius > radius_max:
                new_parameters.radius = radius_max

            return new_parameters
        
        else:
            parameter_estimate = []

            for i in range(len(x_train[0])):

                # Get current independent variables
                current_parameter_set = [elem[i] for elem in x_train]

                # Obtain the parameter in questions upper and lower limits
                p_min = old_parameter.min_max[i][0]
                p_max = old_parameter.min_max[i][1]

                # Fit polynomial to the parameter being modelled
                f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y_train,
                                                                  deg=polynomial_degree, full=False)

                f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly, domain=[p_min, p_max], window=[p_min, p_max])

                # get gradient
                f_poly_deriv = f_poly.deriv()

                current_estimation = self.get_parameter(old_parameter,i)
                
                delta = f_poly_deriv(current_estimation)

                # update parameter
                new_estimation = current_estimation + step_size*delta

                if (new_estimation < p_min):
                    new_estimation = p_min
                if (new_estimation > p_max):
                    new_estimation = p_max

                parameter_estimate.append(new_estimation)
            return Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])

    ####################################################################################################################
    def mean_estimation(self, x_train,y_train):# y_train is weight of parameters which are equal to
        a_data_set = np.transpose(np.array(x_train))
        # print y_train
        if a_data_set != []:
            a_weights = np.array(y_train)
            levels = a_data_set[0, :]
            # ave_level = np.average(levels, weights=a_weights)
            ave_level = np.average(levels)
            angle = a_data_set[1, :]
            # ave_angle = np.average(angle, weights=a_weights)
            ave_angle = np.average(angle)
            radius = a_data_set[2, :]
            # ave_radius = np.average(radius, weights=a_weights)
            ave_radius = np.average(radius)
            new_parameter = Parameter(ave_level, ave_angle, ave_radius)

        # if a_data_set != []:
        #     levels = a_data_set[0, :]
        #     ave_level = np.average(levels)
        #
        #     angle = a_data_set[1, :]
        #     ave_angle = np.average(angle)
        #
        #     radius = a_data_set[2, :]
        #     ave_radius = np.average(radius)
        #
        #     new_parameter = Parameter(ave_level, ave_angle, ave_radius)
            return new_parameter
        else:
            return None

    ####################################################################################################################
    def copy_last_estimation(self, agent_type):
        if agent_type == 'l1':
            return copy(self.l1_estimation.get_last_estimation())
        elif agent_type == 'l2':
            return copy(self.l2_estimation.get_last_estimation())
        elif agent_type == 'f1':
            return copy(self.f1_estimation.get_last_estimation())
        elif agent_type == 'f2':
            return copy(self.f2_estimation.get_last_estimation())
        elif agent_type == 'w':
            return copy(self.w_estimation.get_last_estimation())
        else:
            return None

    ####################################################################################################################
    def parameter_estimation(self, x_train, y_train, agent_type):
        # 1. Getting the last agent parameter estimation
        last_parameters_value = self.copy_last_estimation(agent_type)

        # 2. Running the estimation method if the train data
        # sets are not empty
        if x_train != [] and y_train != []:
            if self.parameter_estimation_mode == 'MIN':
                estimated_parameter = self.mean_estimation(x_train,y_train)
            elif self.parameter_estimation_mode == 'AGA':
                estimated_parameter = self.calculate_gradient_ascent(x_train, y_train, last_parameters_value)
            elif self.parameter_estimation_mode == 'ABU':
                estimated_parameter = self.bayesian_updating(x_train, y_train, last_parameters_value)
            else:
                estimated_parameter = None
        else:
            estimated_parameter = last_parameters_value

        return estimated_parameter

    ####################################################################################################################
    def get_last_selected_type_probability(self,selected_type):
        # TYPE L1 ------------------ 
        if selected_type == 'l1':
            return self.l1_estimation.get_last_type_probability()
        # TYPE L2 ------------------ 
        elif selected_type == 'l2':
            return self.l2_estimation.get_last_type_probability()
        # TYPE F1 ------------------ 
        elif selected_type == 'f1':
            return self.f1_estimation.get_last_type_probability()
        # TYPE F2 ------------------ 
        elif selected_type == 'f2':
            return self.f2_estimation.get_last_type_probability()
        # TYPE W ------------------ 
        elif selected_type == 'w':
            return self.w_estimation.get_last_type_probability()
        else:
            return 0

    ####################################################################################################################
    def update_internal_state(self, parameters_estimation, selected_type, unknown_agent, po):
        # 1. Defining the agent to update in main agent point of view
        u_agent = None 
        if not po:
            u_agent = unknown_agent.choose_target_state.main_agent.visible_agents[unknown_agent.index]
        else:
            memory_agents = unknown_agent.choose_target_state.main_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    u_agent = m_a
                    break

        # 2. Creating the agents for simulation
        tmp_sim = copy(unknown_agent.choose_target_state)
        (x,y), direction = u_agent.get_position(),u_agent.direction 
        tmp_agent = agent.Agent(x,y,direction,selected_type,-1)
        tmp_agent.set_parameters(tmp_sim,parameters_estimation.level,parameters_estimation.radius,parameters_estimation.angle)

        # 3. Finding the target
        tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.agents)
        target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.agents)

        self.iteration += 1
        return target

    ####################################################################################################################
    def copy_train_data(self, selected_type):
        if selected_type == 'l1':
            return copy(self.l1_estimation.train_data)
        elif selected_type == 'l2':
            return copy(self.l2_estimation.train_data)
        elif selected_type == 'f1':
            return copy(self.f1_estimation.train_data)
        elif selected_type == 'f2':
            return copy(self.f2_estimation.train_data)
        elif selected_type == 'w':
            return copy(self.w_estimation.train_data)
        else:
            return None

    ####################################################################################################################
    def get_train_data(self, selected_type):
        if selected_type == 'l1':
            return copy(self.l1_estimation.train_data)
        elif selected_type == 'l2':
            return copy(self.l2_estimation.train_data)
        elif selected_type == 'f1':
            return copy(self.f1_estimation.train_data)
        elif selected_type == 'f2':
            return copy(self.f2_estimation.train_data)
        elif selected_type == 'w':
            return copy(self.w_estimation.train_data)
        else:
            return None

    ####################################################################################################################
    def update_train_data(self, u_a, previous_state, current_state, selected_type, po):
        # 1. Copying the selected type train data
        # unknown_agent = deepcopy(u_a)
        unknown_agent = (u_a)
        train_data = self.get_train_data(selected_type)
        
        # 2. Updating th Particles
        type_probability = self.get_last_selected_type_probability(selected_type)
        if self.train_mode == 'history_based':
            if unknown_agent.next_action == 'L':
                # a. Evaluating the particles
                type_probability = train_data.update_data_set(unknown_agent,current_state,po)
                train_data.generate_data(unknown_agent, selected_type,
                                         self.actions_to_reach_target,
                                         self.action_history)

                # b. Generating
            else:
                train_data.generate_data(unknown_agent,selected_type ,
                                         self.actions_to_reach_target,
                                         self.action_history)

        else:
            self.data = train_data.\
                generate_data_for_update_parameter(previous_state,unknown_agent,selected_type, po)

        # 3. Updating the estimation train data
        if selected_type == 'l1':
            self.l1_estimation.train_data = copy(train_data)

        if selected_type == 'l2':
            self.l2_estimation.train_data = copy(train_data)

        if selected_type == 'f1':
            self.f1_estimation.train_data = copy(train_data)

        if selected_type == 'f2':
            self.f2_estimation.train_data = copy(train_data)

        if selected_type == 'w':
            self.w_estimation.train_data = copy(train_data)
            
        # 4. Extrating and returning the train set
        x_train, y_train = train_data.extract_train_set()
        return x_train, y_train, type_probability

    ####################################################################################################################
    def process_parameter_estimations(self, unknown_agent,previous_state,\
    current_state, enemy_action_prob, types, po = False):
        # 1. Initialising the parameter variables
        x_train, types_train_data = [], []
        new_parameters_estimation = None

        # 2. Estimating the agent type
        for selected_type in types:
            # a. updating the train data for the current state
            x_train, y_train, pf_type_probability = \
            self.update_train_data(unknown_agent,\
                previous_state, current_state, selected_type,po)

            # b. estimating the type with the new train data
            new_parameters_estimation = \
            self.parameter_estimation(x_train, y_train, selected_type)
            
            # c. considering the new estimation
            if new_parameters_estimation is not None:
                # i. generating the particle for the selected type
                if selected_type != 'w':
                    tmp_sim = previous_state.copy()

                    x = unknown_agent.previous_agent_status.position[0] 
                    y = unknown_agent.previous_agent_status.position[1]
                    direction = unknown_agent.previous_agent_status.direction
                    tmp_agent = agent.Agent(x, y, direction, selected_type)

                    tmp_agent.set_parameters(tmp_sim, new_parameters_estimation.level,\
                        new_parameters_estimation.radius,new_parameters_estimation.angle)
                    tmp_agent.memory = self.update_internal_state(new_parameters_estimation,\
                        selected_type,unknown_agent,po)

                    # Runs a simulator object
                    tmp_agent = tmp_sim.move_a_agent(tmp_agent)
                    action_prob = tmp_agent.get_action_probability(unknown_agent.next_action)
                    if action_prob is None:
                        action_prob = 0.2
                    # print action_prob
                    # ii. testing the generated particle and updating the estimation
                    # TYPE L1 ------------------ 
                    if selected_type == 'l1':
                        if self.train_mode == 'history_based':
                            # if unknown_agent.next_action != 'L':
                            #     self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()
                            # else:


                            if self.type_estimation_mode == 'BTE':
                                self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()

                            if self.type_estimation_mode == 'PTE' or self.type_estimation_mode == 'BPTE':
                                self.l1_estimation.type_probability = pf_type_probability

                        else:
                            self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()
                        self.l1_estimation.update_estimation(new_parameters_estimation, action_prob)
                    # TYPE L2 ------------------ 
                    elif selected_type == 'l2':
                        if self.train_mode == 'history_based':
                            # if unknown_agent.next_action != 'L':
                            #     self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()
                            # else:

                            if self.type_estimation_mode == 'BTE':
                                self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()
                            if self.type_estimation_mode == 'BPTE':
                                self.l2_estimation.type_probability = pf_type_probability * self.l2_estimation.get_last_type_probability()
                            if self.type_estimation_mode == 'PTE':
                                self.l2_estimation.type_probability = pf_type_probability
                        else:
                            self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()
                        self.l2_estimation.update_estimation(new_parameters_estimation, action_prob)
                    # TYPE F1 ------------------ 
                    elif selected_type == 'f1':
                        if self.train_mode == 'history_based':
                            if unknown_agent.next_action != 'L':
                                self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()
                            else:

                                if self.type_estimation_mode == 'BTE':
                                    self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()
                                if self.type_estimation_mode == 'BPTE':
                                    self.f1_estimation.type_probability = pf_type_probability * self.f1_estimation.get_last_type_probability()
                                if self.type_estimation_mode == 'PTE':
                                    self.f1_estimation.type_probability = pf_type_probability
                        else:
                            self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()
                        self.f1_estimation.update_estimation(new_parameters_estimation, action_prob)
                    # TYPE F2 ------------------ 
                    elif selected_type == 'f2':
                        if self.train_mode == 'history_based':
                            if unknown_agent.next_action != 'L':
                                self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()
                            else:
                                if self.type_estimation_mode == 'BTE':
                                    self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()
                                if self.type_estimation_mode == 'BPTE':
                                    self.f2_estimation.type_probability = pf_type_probability * self.f2_estimation.get_last_type_probability()
                                if self.type_estimation_mode == 'PTE':
                                    self.f2_estimation.type_probability = pf_type_probability
                        else:
                            self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()
                        self.f2_estimation.update_estimation(new_parameters_estimation, action_prob)
                # ADVERSARY ------------------ 
                else:
                    if self.train_mode == 'history_based':
                        self.w_estimation.type_probability = enemy_action_prob * self.w_estimation.get_last_type_probability()
                    else:
                        self.w_estimation.type_probability = enemy_action_prob * self.w_estimation.get_last_type_probability()
                    self.w_estimation.update_estimation(new_parameters_estimation, enemy_action_prob)

        # d. If a load action was performed, restart the estimation process
        if unknown_agent.next_action == 'L' and unknown_agent.is_item_nearby(current_state.items) != -1:
            if unknown_agent.choose_target_state != None:
                hist = {}
                hist['pos'] = copy(unknown_agent.choose_target_pos)
                hist['direction'] = unknown_agent.choose_target_direction

                hist['state'] = unknown_agent.choose_target_state.copy()  # todo: replace it with items and agents position instead of whole state!
                hist['loaded_item'] = copy(unknown_agent.last_loaded_item_pos)
                unknown_agent.choose_target_history.append(hist)


            unknown_agent.choose_target_state = current_state.copy()
            unknown_agent.choose_target_pos = unknown_agent.get_position()
            unknown_agent.choose_target_direction = unknown_agent.direction

        # e. Normalising the type probabilities
        if self.train_mode == 'history_based':
            if self.type_estimation_mode == 'BPTE':
                self.l1_estimation.type_probability = self.normalize_type_probability('l1') * self.l1_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.normalize_type_probability('l2') * self.l2_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.normalize_type_probability('f1') * self.f1_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.normalize_type_probability('f2') * self.f2_estimation.get_last_type_probability()
            self.alpha = 0.1
            if self.type_estimation_mode == 'LPTE':
                self.l1_estimation.type_probability = self.alpha *  self.normalize_type_probability('l1') + \
                                                      (1-self.alpha) * self.l1_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.alpha *  self.normalize_type_probability('l2') + \
                                                      (1-self.alpha) * self.l2_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.alpha *  self.normalize_type_probability('f1') + \
                                                      (1-self.alpha) * self.f1_estimation.get_last_type_probability()
                self.l1_estimation.type_probability = self.alpha *  self.normalize_type_probability('f2') + \
                                                      (1-self.alpha) * self.f2_estimation.get_last_type_probability()
        self.normalize_type_probabilities()
        print '>>> %d) %.4lf %.4lf %.4lf %.4lf' %(unknown_agent.index,self.l1_estimation.type_probability,self.l2_estimation.type_probability, \

        self.f1_estimation.type_probability,self.f2_estimation.type_probability)

    ####################################################################################################################
    def unseen_parameter_estimation_not_update(self,unknown_agent,types):
        # 1. Repeting the last estimation for all types and do not updating
        # the type probability
        for selected_type in types:
            # TYPE L1 ------------------ 
            if selected_type == 'l1':
                new_parameters_estimation = self.l1_estimation.get_last_estimation()
                action_prob = self.l1_estimation.get_last_action_probability()
                self.l1_estimation.type_probability = self.l1_estimation.get_last_type_probability()
                self.l1_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE L2 ------------------ 
            elif selected_type == 'l2':
                new_parameters_estimation = self.l2_estimation.get_last_estimation()
                action_prob = self.l2_estimation.get_last_action_probability()
                self.l2_estimation.type_probability = self.l2_estimation.get_last_type_probability()
                self.l2_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE F1 ------------------ 
            elif selected_type == 'f1':
                new_parameters_estimation = self.f1_estimation.get_last_estimation()
                action_prob = self.f1_estimation.get_last_action_probability()
                self.f1_estimation.type_probability = self.f1_estimation.get_last_type_probability()
                self.f1_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE F2 ------------------ 
            elif selected_type == 'f2':
                new_parameters_estimation = self.f2_estimation.get_last_estimation()
                action_prob = self.f2_estimation.get_last_action_probability()
                self.f2_estimation.type_probability = self.f2_estimation.get_last_type_probability()
                self.f2_estimation.update_estimation(new_parameters_estimation, action_prob)
            # ADVERSARY ------------------ 
            elif selected_type == 'w':
                new_parameters_estimation = self.w_estimation.get_last_estimation()
                enemy_action_prob = self.w_estimation.get_last_action_probability()
                self.w_estimation.type_probability = self.w_estimation.get_last_type_probability()
                self.w_estimation.update_estimation(new_parameters_estimation, enemy_action_prob)
        # 2. Appending type probabilities
        self.normalize_type_probabilities()
        print '>>> %d) %.4lf %.4lf %.4lf %.4lf' %(unknown_agent.index,self.l1_estimation.type_probability,self.l2_estimation.type_probability,\
        self.f1_estimation.type_probability,self.f2_estimation.type_probability)

    ####################################################################################################################
    def unseen_parameter_estimation_particle_evaluation(self,state,u_a,types):
        # 1. Updating with the uniform action probability
        for selected_type in types:
            # a. copying the simulator
            tmp_sim = state.simulator.copy()

            # b. getting the agent
            tmp_agent = None
            for agent in tmp_sim.agents:
                if agent.index == u_a.index:
                    tmp_agent = deepcopy(agent)
                    break

            # a. getting the train data and extracting it
            train_data = self.get_train_data(selected_type) 
            x_train, y_train = train_data.extract_train_set()

            # b. estimating the type with the new train data
            new_parameters_estimation = \
            self.parameter_estimation(x_train, y_train, selected_type)

            tmp_agent.agent_type = selected_type

            if new_parameters_estimation is None:
                if self.train_mode == 'history_based':
                    particle = random.sample(train_data.data_set,1)[0]
                    tmp_level = particle['parameter'][0]
                    tmp_radius = particle['parameter'][1]
                    tmp_angle = particle['parameter'][2]
                else:
                    i = random.randint(0,train_data.generated_data_number)
                    tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / train_data.generated_data_number) * i
                    tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / train_data.generated_data_number) * i
                    tmp_level = level_min + (1.0 * (level_max - level_min) / train_data.generated_data_number) * i 
                
                new_parameters = Parameter(tmp_level,tmp_angle,tmp_radius)
            else:
                tmp_radius = new_parameters_estimation.level
                tmp_angle = new_parameters_estimation.angle
                tmp_level = new_parameters_estimation.radius

            tmp_agent.set_parameters(tmp_sim, new_parameters_estimation.level,\
                    new_parameters_estimation.radius,new_parameters_estimation.angle)

            # Runs a simulator object
            tmp_agent = tmp_sim.move_a_agent(tmp_agent)
            if tmp_agent.next_action is not None:
                action_prob = tmp_agent.get_action_probability(tmp_agent.next_action)
            else:
                action_prob = 0.2
            # print '>>>>',action_prob

            # TYPE L1 ------------------ 
            if selected_type == 'l1':
                new_parameters_estimation = self.l1_estimation.get_last_estimation()
                self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()
                self.l1_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE L2 ------------------ 
            elif selected_type == 'l2':
                new_parameters_estimation = self.l2_estimation.get_last_estimation()
                self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()
                self.l2_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE F1 ------------------ 
            elif selected_type == 'f1':
                new_parameters_estimation = self.f1_estimation.get_last_estimation()
                self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()
                self.f1_estimation.update_estimation(new_parameters_estimation, action_prob)
            # TYPE F2 ------------------ 
            elif selected_type == 'f2':
                new_parameters_estimation = self.f2_estimation.get_last_estimation()
                self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()
                self.f2_estimation.update_estimation(new_parameters_estimation, action_prob)
            # ADVERSARY ------------------ 
            elif selected_type == 'w':
                new_parameters_estimation = self.w_estimation.get_last_estimation()
                self.w_estimation.type_probability = action_prob * self.w_estimation.get_last_type_probability()
                self.w_estimation.update_estimation(new_parameters_estimation,action_prob)
        
        # 2. Appending type probabilities
        self.normalize_type_probabilities()
        print '>>> %d) %.4lf %.4lf %.4lf %.4lf' %(u_a.index,self.l1_estimation.type_probability,self.l2_estimation.type_probability,\
        self.f1_estimation.type_probability,self.f2_estimation.type_probability)