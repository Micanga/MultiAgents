import random
from numpy.random import choice
from sklearn import linear_model
import numpy as np
import scipy.stats as st
import agent

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

types = ['l1', 'l2', 'f1', 'f2', 'w']

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
    def __init__(self, a_type, generated_data_number, PF_add_threshold, train_mode):
        self.type = a_type  # Type for which we are doing estimation
        self.type_probability = 0
        self.type_probabilities = []
        self.estimation_history = []
        self.action_probabilities = []
        self.internal_state = None
        self.train_mode = train_mode
        self.train_data = train_data.TrainData(generated_data_number, PF_add_threshold,  train_mode)

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
        return self.type_probabilities[len(self.type_probabilities) - 1]

    ####################################################################################################################
    def get_last_estimation(self):
        return self.estimation_history[len(self.estimation_history) - 1]

    ####################################################################################################################
    def update_estimation(self, estimation, action_probability):
        self.estimation_history.append(estimation)
        self.action_probabilities.append(action_probability)


########################################################################################################################
class ParameterEstimation:

    def __init__(self,  generated_data_number, PF_add_threshold, train_mode,apply_adversary):

        # P(teta|H)
        self.apply_adversary = apply_adversary
        if self.apply_adversary:
            self.w_estimation = TypeEstimation('w', generated_data_number, PF_add_threshold, train_mode)

        self.l1_estimation = TypeEstimation('l1', generated_data_number, PF_add_threshold, train_mode)
        self.l2_estimation = TypeEstimation('l2',  generated_data_number, PF_add_threshold, train_mode)
        self.f1_estimation = TypeEstimation('f1',  generated_data_number, PF_add_threshold, train_mode)
        self.f2_estimation = TypeEstimation('f2',  generated_data_number, PF_add_threshold, train_mode)


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

    def estimation_configuration(self, type_selection_mode, parameter_estimation_mode, polynomial_degree):

        self.type_selection_mode = type_selection_mode
        self.parameter_estimation_mode = parameter_estimation_mode
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

    ####################################################################################################################

    def update_internal_state(self, parameters_estimation, selected_type, uknown_agent):

        u_agent = uknown_agent.choose_target_state.main_agent.visible_agents[uknown_agent.index]

        tmp_sim = uknown_agent.choose_target_state
        (x, y) = u_agent.get_position()

        tmp_agent = agent.Agent(x, y, u_agent.direction, selected_type, -1)

        tmp_agent.set_parameters(uknown_agent.choose_target_state, parameters_estimation.level, parameters_estimation.radius,
                                 parameters_estimation.angle)

        # find the target with
        tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.agents)
        target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.agents)
        self.iteration += 1
        return target
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

            #print('Parameter Estimate: {}'.format(parameter_estimate))
            
            return Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
        
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
                    print('Proposal: {}'.format(proposal))
                    if proposal > polynomial_max:
                        polynomial_max = proposal

                parameter_estimate.append(polynomial_max)

            elif sampling == 'average':
                x_random = self.sampleFromBelief(new_belief, 10)
                parameter_estimate.append(np.mean(x_random))

            # Increment iterator

        new_parameter = Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
        print('Parameter Estimate: {}'.format(parameter_estimate))
        self.iteration += 1

        return new_parameter

    ####################################################################################################################
    def update_action_history(self, current_action):

        self.action_history.append(current_action)
        if current_action == 'L':
            self.actions_to_reach_target = []
        else:
            self.actions_to_reach_target.append(current_action)

    ####################################################################################################################

    def update_train_data(self, unknown_agent, previous_state, current_state, selected_type ,po= False):

        tmp_train_data = None

        if selected_type == 'l1':
            tmp_train_data = copy(self.l1_estimation.train_data)

        if selected_type == 'l2':
            tmp_train_data = copy(self.l2_estimation.train_data)

        if selected_type == 'f1':
            tmp_train_data = copy(self.f1_estimation.train_data)

        if selected_type == 'f2':
            tmp_train_data = copy(self.f2_estimation.train_data)

        if selected_type == 'w':
            tmp_train_data = copy(self.w_estimation.train_data)

        max_succeeded_steps = 0
        if self.train_mode == 'history_based':

            if unknown_agent.next_action == 'L':
                print '      ******* Update data set _ deep copy *******'
                unknown_agent.choose_target_state =copy( current_state)
                print '      ******* train data *******'
                max_succeeded_steps = tmp_train_data.update_data_set(unknown_agent, self.actions_to_reach_target, selected_type , po)
                print '      ******* after  train data *******'
            else:
                max_succeeded_steps = tmp_train_data.generate_data(unknown_agent,
                                             self.action_history,
                                             self.actions_to_reach_target,
                                             selected_type )

        else:  # Not history_based
            self.data = tmp_train_data.generate_data_for_update_parameter\
                (previous_state,
                 unknown_agent,
                 selected_type,po)

        if selected_type == 'l1':
            self.l1_estimation.train_data = copy(tmp_train_data)

        if selected_type == 'l2':
            self.l2_estimation.train_data = copy(tmp_train_data)

        if selected_type == 'f1':
            self.f1_estimation.train_data = copy(tmp_train_data)

        if selected_type == 'f2':
            self.f2_estimation.train_data = copy(tmp_train_data)

        if selected_type == 'w':
            self.w_estimation.train_data = copy(tmp_train_data)

        x_train, y_train = tmp_train_data.extract_train_set()
        return x_train, y_train, max_succeeded_steps

    ####################################################################################################################
    def parameter_estimation(self, x_train, y_train, agent_type):

        estimated_parameter = None

        last_parameters_value = 0

        if agent_type == 'l1':
            last_parameters_value = copy(self.l1_estimation.get_last_estimation())
            # x_train, y_train = self.l1_estimation.train_data.get_data_set(sim, cur_agent, action)

        if agent_type == 'l2':
            last_parameters_value = copy(self.l2_estimation.get_last_estimation())
            # x_train, y_train = self.l2_estimation.train_data.get_data_set(sim, cur_agent, action)

        if agent_type == 'f1':
            last_parameters_value = copy(self.f1_estimation.get_last_estimation())
            # x_train, y_train = self.f1_estimation.train_data.get_data_set(sim, cur_agent, action)

        if agent_type == 'f2':
            last_parameters_value = copy(self.f2_estimation.get_last_estimation())
            # x_train, y_train = self.f2_estimation.train_data.get_data_set(sim, cur_agent, action)

        if agent_type == 'w':
            last_parameters_value = copy(self.w_estimation.get_last_estimation())
            # x_train, y_train = self.w_estimation.train_data.get_data_set(sim, cur_agent, action)

        if x_train != [] and y_train != []:
            if self.parameter_estimation_mode == 'MIN':
                estimated_parameter = self.mean_estimation(x_train)

            # D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)
            if self.parameter_estimation_mode == 'AGA':
                estimated_parameter = self.calculate_gradient_ascent(x_train, y_train, last_parameters_value)

            if self.parameter_estimation_mode == 'ABU':
                estimated_parameter = self.bayesian_updating(x_train, y_train, last_parameters_value)
        else:
            estimated_parameter = last_parameters_value

        return estimated_parameter

    ####################################################################################################################
    def mean_estimation(self, x_train):
        # parameters = []update_train_data
        # for ds in x_train:
        #     parameters.append(ds)

        # print parameters
        a_data_set = np.transpose(np.array(x_train))

        if a_data_set != []:
            # a_weights = np.array(current_weight)
            #
            levels = a_data_set[0, :]
            ave_level = np.average(levels)  # , weights=a_weights)
            #
            angle = a_data_set[1, :]
            ave_angle = np.average(angle)  # , weights=a_weights)
            #
            radius = a_data_set[2, :]
            ave_radius = np.average(radius)  # , weights=a_weights)
            new_parameter = Parameter(ave_level, ave_angle, ave_radius)
            # print 'new_parameter', ave_level, ave_angle, ave_radius
            # new_parameter = deepcopy(self.l1_estimation.get_last_estimation())

            return new_parameter
        else:
            return None

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

        print('UCB Algorithm returned agent of type: {}'.format(return_agent[0]))

        if final:
            return return_agent
        else:
            return ['f2']


    ####################################################################################################################
    def update_internal_state(self, parameters_estimation, selected_type, uknown_agent,po = False):

        if po:# Partial Observation
            u_agent = None
            mem_agents = uknown_agent.choose_target_state.main_agent.agent_memory
            for m_a in mem_agents:
                if m_a.index == uknown_agent.index :
                    u_agent = m_a
        else:
            u_agent = uknown_agent.choose_target_state.main_agent.visible_agents[uknown_agent.index]

        tmp_sim = uknown_agent.choose_target_state
        (x,y) = u_agent.get_position()

        tmp_agent = agent.Agent(x,y,u_agent.direction,selected_type,-1)

        tmp_agent.set_parameters(uknown_agent.choose_target_state,
                                 parameters_estimation.level,
                                 parameters_estimation.radius,
                                 parameters_estimation.angle)

        # find the target with
        tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.agents)
        target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.agents)
        self.iteration += 1
        return target

    ####################################################################################################################

    def process_parameter_estimations(self, time_step, unknown_agent,previous_state, current_state,enemy_action_prob, po=False,actions = None):
        # Initialising the parameter variables
        x_train = []

        types_train_data = []
        new_parameters_estimation = None
        selected_types = None

        # Start parameter estimation
        if self.type_selection_mode == 'AS':
            selected_types = types
        if self.type_selection_mode == 'BS':
            selected_types = self.UCB_selection(time_step)  # returns l1, l2, f1, f2,w

        if actions == 1:
            unknown_agent.next_action = 'L'
        print ('unknown_agent.next_action=',unknown_agent.next_action)
        if self.train_mode == 'history_based':
            self.action_history.append(unknown_agent.next_action)
            if unknown_agent.next_action != 'L':
                self.actions_to_reach_target.append(unknown_agent.next_action)

        print self.actions_to_reach_target
        if self.apply_adversary:
            selected_types = ['l1', 'l2','w']
        else:
            selected_types = ['l1']

        if unknown_agent.next_action is None:
            return

        # Estimate the parameters
        # print 'action', unknown_agent.next_action
        for selected_type in selected_types:
            # Generates an Agent object
            # print('Selected Type'), selected_type, ' --------------------'
            #
            # =============================== Create new estimation ====================================================
            # type_probability is only used tor history based
            # print 'Train data for ', unknown_agent.index
            x_train, y_train, type_probability = \
                self.update_train_data(unknown_agent, previous_state,
                                       current_state, selected_type,po)
            #
            # type_train_data = {}
            # type_train_data['type'] = selected_type
            # type_train_data['x_train'] = x_train
            # types_train_data.append(type_train_data)

            # print 'Estimating parameters for agent', unknown_agent.index
            new_parameters_estimation = self.parameter_estimation(x_train, y_train, selected_type)
            
            print 'new estimated parameters:'\
                , str(new_parameters_estimation.level)\
                , str(new_parameters_estimation.radius)\
                , str(new_parameters_estimation.angle)

            # ==========================================================================================================

            # ===== moving temp agent in last step map with new parameters =============================================
            if new_parameters_estimation is not None:
                if selected_type != 'w':

                    x,y = unknown_agent.previous_agent_status.get_position()
                    tmp_agent = agent.Agent(x, y, unknown_agent.previous_agent_status.direction, selected_type)
                    tmp_agent.memory = self.update_internal_state(new_parameters_estimation, selected_type, unknown_agent,po)

                    tmp_agent.set_parameters(previous_state, new_parameters_estimation.level,
                                             new_parameters_estimation.radius,
                                             new_parameters_estimation.angle)

                    # Runs a simulator object
                    tmp_agent = previous_state.move_a_agent(tmp_agent)

                    action_prob = tmp_agent.get_action_probability(unknown_agent.next_action)

                if selected_type == 'l1':

                    # print 'Last type probability', self.l1_estimation.get_last_type_probability()
                    if self.train_mode == 'history_based':
                        self.l1_estimation.type_probability = action_prob * type_probability
                        # self.l1_estimation.get_last_type_probability()

                    else:
                        # print self.l1_estimation.get_last_type_probability()
                        # print action_prob
                        self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()

                    # print 'type prob:', self.l1_estimation.type_probability
                    self.l1_estimation.update_estimation(new_parameters_estimation, action_prob)
                    # print 'New type probability', self.l1_estimation.type_probability

                if selected_type == 'l2':
                    # print 'Last type probability', self.l2_estimation.get_last_type_probability()
                    if self.train_mode == 'history_based':
                        self.l2_estimation.type_probability = action_prob * type_probability
                        # type_probability *
                    else:
                        print self.l2_estimation.get_last_type_probability()
                        print action_prob
                        self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()

                    # print 'type prob:', self.l2_estimation.type_probability
                    self.l2_estimation.update_estimation(new_parameters_estimation, action_prob)
                    # print 'New type probability', self.l2_estimation.type_probability

                if selected_type == 'f1':

                    if self.train_mode == 'history_based':
                        self.l1_estimation.type_probability = action_prob * \
                                                              self.f1_estimation.get_last_type_probability()
                        # type_probability *
                    else:
                        self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()
                    # print 'f1', self.f1_estimation.train_data.data_set
                    self.f1_estimation.update_estimation(new_parameters_estimation, action_prob)

                if selected_type == 'f2':

                    if self.train_mode == 'history_based':
                        self.l1_estimation.type_probability = action_prob * \
                                                              self.f2_estimation.get_last_type_probability()
                        # type_probability *
                    else:
                        self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()
                    # print 'f2', self.f2_estimation.train_data.data_set

                    self.f2_estimation.update_estimation(new_parameters_estimation, action_prob)
                if selected_type == 'w':

                    if self.train_mode == 'history_based':
                        self.w_estimation.type_probability = enemy_action_prob * \
                                                              self.w_estimation.get_last_type_probability()
                        # type_probability *
                    else:
                        self.w_estimation.type_probability = enemy_action_prob * self.w_estimation.get_last_type_probability()
                    # print 'f2', self.f2_estimation.train_data.data_set

                    self.w_estimation.update_estimation(new_parameters_estimation, enemy_action_prob)

        if unknown_agent.next_action == 'L':
            self.actions_to_reach_target = []

        self.normalize_type_probabilities()

        return new_parameters_estimation,x_train

    ####################################################################################################################
    def normalize_type_probabilities(self):

        l1_update_belief_value = self.l1_estimation.type_probability
        l2_update_belief_value = self.l2_estimation.type_probability
        f1_update_belief_value = self.f1_estimation.type_probability
        f2_update_belief_value = self.f2_estimation.type_probability

        if self.apply_adversary:
            w_update_belief_value = self.w_estimation.type_probability

        sum_of_probabilities = l1_update_belief_value + l2_update_belief_value + \
                               f1_update_belief_value + f2_update_belief_value

        if self.apply_adversary :
            sum_of_probabilities += w_update_belief_value

        belief_factor = 1

        if sum_of_probabilities != 0:
            belief_factor = 1 / sum_of_probabilities

            w_prob = 0
            l1_prob = l1_update_belief_value * belief_factor
            l2_prob = l2_update_belief_value * belief_factor
            f1_prob = f1_update_belief_value * belief_factor
            f2_prob = f2_update_belief_value * belief_factor

            if self.apply_adversary:
                w_prob = w_update_belief_value * belief_factor

            self.l1_estimation.type_probabilities.append(l1_prob)
            self.l2_estimation.type_probabilities.append(l2_prob)
            self.f1_estimation.type_probabilities.append(f1_prob)
            self.f2_estimation.type_probabilities.append(f2_prob)

            if self.apply_adversary:
                self.w_estimation.type_probabilities.append(w_prob)
        else:
            if self.apply_adversary:
                self.l1_estimation.type_probabilities.append(0.2)
                self.l2_estimation.type_probabilities.append(0.2)
                self.f1_estimation.type_probabilities.append(0.2)
                self.f2_estimation.type_probabilities.append(0.2)
                self.w_estimation.type_probabilities.append(0.2)
            else:
                self.l1_estimation.type_probabilities.append(0.25)
                self.l2_estimation.type_probabilities.append(0.25)
                self.f1_estimation.type_probabilities.append(0.25)
                self.f2_estimation.type_probabilities.append(0.25)

    ####################################################################################################################
    def set_choose_target_state(self, state, agent_type):

        if agent_type == 'l1':
            self.l1_estimation.choose_target_state = state

        if agent_type == 'l2':
            self.l2_estimation.choose_target_state = state

        if agent_type == 'f1':
            self.f1_estimation.choose_target_state = state

        if agent_type == 'f2':
            self.f2_estimation.choose_target_state = state

        if agent_type == 'w':
            self.w_estimation.choose_target_state = state

####################################################################################################################
    def plot_data_set(self):

         # fig = plt.figure(1)
      #   print self.l1_estimation.type_probabilities
      #   # plt.plot([i for i in range(len(self.l1_estimation.type_probabilities))],
      #   #          self.l1_estimation.type_probabilities,
      #   #          label='l1 probability',
      #   #          linestyle='-',
      #   #          color='cornflowerblue',
      #   #          linewidth=1)
      #   plt.plot( self.l2_estimation.type_probabilities )
      #   # ax = plt.gca()
      #   # plt.set_ylabel('Type Probability')
      #   # ax.legend(loc="upper right", shadow=True, fontsize='x-large')
      #   # plt.subplot(3, 1, 2)
      #
      #
      # #  fig.savefig("./plots/type_probability_changes.jpg")
      #   plt.show()

        fig = plt.figure(1)
        plt.subplot(3, 1, 1)

        plt.plot(self.l1_estimation.type_probabilities)
        ax = plt.gca()

        ax.set_ylabel('L1   Probabilities')
     #   ax.legend(loc="upper right", shadow=True, fontsize='x-large')
        plt.subplot(3, 1, 2)

        plt.plot(self.l2_estimation.type_probabilities)
        ax = plt.gca()
        ax.set_ylabel('L2   Probabilities')

        plt.subplot(3, 1, 3)

        plt.plot(self.f1_estimation.type_probabilities)

        ax = plt.gca()
        ax.set_ylabel('f1   Probabilities')
        ax.set_xlabel('iteration')


        #fig.savefig("./plots/dataset_history_based.jpg")
        plt.show()
