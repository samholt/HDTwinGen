def system_prompt():
    return '''
Objective: Write code to create an effective differential equation simulator for a given task.
Please note that the code should be fully functional. No placeholders.

You must act autonomously and you will receive no human input at any stage. You have to return as output the complete code for completing this task, and correctly improve the code to create the most accurate and realistic simulator possible.
You always write out the code contents. You always indent code with tabs.
You cannot visualize any graphical output. You exist within a machine. The code can include black box multi-layer perceptions where required.

Use the functions provided. When calling functions only provide a RFC8259 compliant JSON request following this format without deviation.
'''

def get_system_description(env_name):
    if env_name == 'Cancer' or 'Cancer-random' in env_name:
        return """Prediction of Treatment Response for Combined Chemo and Radiation Therapy for Non-Small Cell Lung Cancer Patients Using a Bio-Mathematical Model

Here you must model the state differential of tumor_volume, and chemotherapy_drug_concentration; with the input actions of chemotherapy_dosage, and radiotherapy_dosage.

Description of the variables:
* tumor_volume: Volume of the tumor with units cm^3
* chemotherapy_drug_concentration: Concentration of the chemotherapy drug vinblastine with units mg/m^3
* chemotherapy_dosage: Dosage of the chemotherapy drug vinblastine with units mg/m^3
* radiotherapy_dosage: Dosage of the radiotherapy with units Gy

The time units is in days.

Additionally these variables have the ranges of:
* tumor_volume: [0.01433, 1170.861]
* chemotherapy_drug_concentration: [0, 9.9975]
* chemotherapy_dosage: [0, 5.0]
* radiotherapy_dosage: [0, 2.0]

The training dataset consists of 1000 patients, where each patient is observed for 60 days."""
    elif env_name == 'Cancer-ood' or env_name == 'Cancer-iid':
        return """Prediction of Treatment Response for Combined Chemo and Radiation Therapy for Non-Small Cell Lung Cancer Patients Using a Bio-Mathematical Model

Here you must model the state differential of tumor_volume, and chemotherapy_drug_concentration; with the input actions of chemotherapy_dosage, and radiotherapy_dosage.

Description of the variables:
* tumor_volume: Volume of the tumor with units cm^3
* chemotherapy_drug_concentration: Concentration of the chemotherapy drug vinblastine with units mg/m^3
* chemotherapy_dosage: Dosage of the chemotherapy drug vinblastine with units mg/m^3
* radiotherapy_dosage: Dosage of the radiotherapy with units Gy

The time units is in hours.

Additionally these variables have the ranges of:
* tumor_volume: [0.01433, 1170.861]
* chemotherapy_drug_concentration: [0, 9.9975]
* chemotherapy_dosage: [0, 5.0]
* radiotherapy_dosage: [0, 2.0]

The training dataset consists of 1000 patients, where each patient is observed for 60 hours."""
    elif env_name == 'Cancer-untreated':
        return """Prediction of Treatment Response for Combined Chemo and Radiation Therapy for Non-Small Cell Lung Cancer Patients Using a Bio-Mathematical Model

Here you must model the state differential of tumor_volume. There are not treatments applied.

Description of the variables:
* tumor_volume: Volume of the tumor with units cm^3

The time units is in days.

Additionally these variables have the ranges of:
* tumor_volume: [0.64196031, 4852.45734281]

The training dataset consists of 1000 patients, where each patient is observed for 60 days."""
    elif env_name == 'Cancer-chemo':
        return """Prediction of Treatment Response for Combined Chemo and Radiation Therapy for Non-Small Cell Lung Cancer Patients Using a Bio-Mathematical Model

Here you must model the state differential of tumor_volume, and chemotherapy_drug_concentration; with the input actions of chemotherapy_dosage.

Description of the variables:
* tumor_volume: Volume of the tumor with units cm^3
* chemotherapy_drug_concentration: Concentration of the chemotherapy drug vinblastine with units mg/m^3
* chemotherapy_dosage: Dosage of the chemotherapy drug vinblastine with units mg/m^3

The time units is in days.

Additionally these variables have the ranges of:
* tumor_volume: [0.64196031, 1260.60290569]
* chemotherapy_drug_concentration: [0, 9.9975]
* chemotherapy_dosage: [0, 5.0]

The training dataset consists of 1000 patients, where each patient is observed for 60 days."""
    elif env_name == 'Dataset-3DLV':
        return """"Modeling Artificial Tri-Trophic Prey-Predator Oscillations in a Simplified Ecological System

Here you must model the state differential of algae_population, flagellate_population, and rotifer_population; with no input actions. This aims to simulate the population dynamics within a simplified tri-trophic ecological system comprising prey (algae), intermediate predators (flagellates), and top predators (rotifers). The interactions include direct predation and competition for resources, mirroring natural intraguild predation mechanisms.

Description of the variables:
* prey_population: Total count of algae, serving as the primary prey
* intermediate_population: Total count of flagellates, acting as intermediate predators and prey
* top_predators_population: Total count of rotifers, representing top predators

The dataset encapsulates daily population counts across multiple simulated ecosystems over a period of 100 days, allowing for the analysis of temporal oscillations and phase lags between species.

Additionally these variables have the ranges of:
* prey_population: [0.095898, 2.469735]
* intermediate_population: [0.008438, 1.500000]
* top_predators_population: [0.030316, 0.739244]

The training dataset consists of 70 time steps, validation and training dataset consists of 15 time steps each.
"""
    elif env_name == 'Dataset-HL':
        return """"Modeling Di-Trophic Prey-Predator Dynamics in a Hare and Lynx Ecological System

Here you must model the state differential of hare_population, and lynx_population; with the additional input of time_in_years. This aims to simulate the population dynamics within a simplified di-trophic ecological system comprising prey (hares), and predators (lynxes). The interactions include direct predation and competition for resources, mirroring natural predator-prey mechanisms.

Description of the variables:
* hare_population: Annual count of hare pelts, serving as a proxy for the hare population size, in tens of thousands.
* lynx_population: Annual count of lynx pelts, serving as a proxy for the lynx population size, in tens of thousands.

The model should capture the dynamics of these populations, reflecting the di-trophic prey-predator interactions, and predict the population sizes based on historical data. The data exhibits 10-year long characteristic oscillations due to prey-predator dynamics.

Additionally these variables have the ranges of:
* hare_population: [1.80, 152.65]
* lynx_population: [3.19, 79.35]
* time_in_years: [1845, 1935]

The training dataset consists of 63 time steps, validation and training dataset consists of 14 time steps each.
"""
    if env_name == 'COVID':
        return """Prediction model of COVID-19 Epidemic Dynamics

Here you must model the state differential of susceptible, exposed, infected and recovered; with the input action of a constant total_population. There are no interventions applied. Here the states are normalized ratios of the total fixed population.

Description of the variables:
* susceptible: Ratio of the population that is susceptible to the virus. 
* exposed: Ratio of the population that is exposed to the virus, not yet infectious.
* infected: Ratio of the population that is actively carrying and transmitting the virus.
* recovered: Ratio of the population that have recovered from the virus, including those who are deceased.
* total_population: Total population of the country, a constant.

The time units is in days.

Additionally these variables have the ranges of:
* susceptible: [0, 1]
* exposed: [0, 1]
* infected: [0, 1]
* recovered: [0, 1]
* total_population: [10000, 10000]

The training dataset consists of 24 countries, where each country is observed for 60 days."""
    else:
        raise NotImplementedError


def get_skeleton_code(env_name):
    if env_name == 'Cancer' or env_name == 'Cancer-ood' or env_name == 'Cancer-iid' or 'Cancer-random' in env_name:
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, tumor_volume: torch.Tensor, chemotherapy_drug_concentration: torch.Tensor, chemotherapy_dosage: torch.Tensor, radiotherapy_dosage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Fill in the code here
        return (d_tumor_volume__dt, d_chemotherapy_drug_concentration__dt)"""
    elif env_name == 'Cancer-chemo':
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, tumor_volume: torch.Tensor, chemotherapy_drug_concentration: torch.Tensor, chemotherapy_dosage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Fill in the code here
        return (d_tumor_volume__dt, d_chemotherapy_drug_concentration__dt)"""
    elif env_name == 'Cancer-untreated':
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, tumor_volume: torch.Tensor) -> Tuple[torch.Tensor]:
        # TODO: Fill in the code here
        return (d_tumor_volume__dt)"""
    elif env_name == 'Dataset-3DLV':
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, prey_population: torch.Tensor, intermediate_population: torch.Tensor, top_predators_population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Fill in the code here
        return (d_prey_population__dt, d_intermediate_population__dt, d_top_predators_population__dt)"""
    elif env_name == 'Dataset-HL':
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, hare_population: torch.Tensor, lynx_population: torch.Tensor, time_in_years: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Fill in the code here
        return (d_hare_population__dt, d_lynx_population__dt)"""
    if env_name == 'COVID':
        return """class StateDifferential(nn.Module):
    def __init__(self):
        super(StateDifferential, self).__init__()
        # TODO: Fill in the code here

    def forward(self, susceptible: torch.Tensor, exposed: torch.Tensor, infected: torch.Tensor, recovered: torch.Tensor, total_population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Fill in the code here
        return (d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt)"""
    else:
        raise NotImplementedError

def first_task_prompt(env_name, generations=20, current_iteration=0):
    system_description = get_system_description(env_name)
    skeleton_code = get_skeleton_code(env_name)
    return f"""
You will get a system description to code a differential equation simulator for.

System Description:```
{system_description}
```

Modelling goals:```
* The parameters of the model will be optimized to an observed training dataset with the given simulator.
* The observed training dataset has very few samples, and the model must be able to generalize to unseen data.
```

Requirement Specification:```
* The code generated should achieve the lowest possible validation loss, of 1e-10 or less.
* The code generated should be interpretable, and fit the dataset as accurately as possible.
```

Skeleton code to fill in:```
{skeleton_code}
```

Useful to know:```
* You are a code evolving machine, and you will be called {generations} times to generate code, and improve the code to achieve the lowest possible validation loss.
* The model defines the state differential and will be used with an ODE solver to fit the observed training dataset.
* You can use any parameters you want and any black box neural network components (multi-layer perceptrons); however, you have to define these.
* It is preferable to decompose the system into differential equations (compartments) if possible.
* You can use any unary functions, for example log, exp, power etc.
* Under no circumstance can you change the skeleton code function definitions, only fill in the code.
* The input tensors are vectors of shape (batch_size).
* Use initially white box models first and then switch to hybrid white and black box models for the residuals, only after no further best program iteration improvement with white box models.
* Make sure your code follows the exact code skeleton specification.
* Use PyTorch.
```
                                        
Think step-by-step, and then give the complete full working code. You are generating code for iteration {current_iteration} out of {generations}.
"""

#         elif self.env.env_name == 'COVID-schools-and-social-distancing':
#             # SEIRD model
#             low_range = self.env.train_data[0].min((0,1))
#             high_range = self.env.train_data[0].max((0,1))
#             initial_prompt = f"""
# You will get a environment task description to code a differential equation simulator for.


# Environment Task description:```
# A COVID-19 model for a country including interventions.
# ```
# You will complete the following function definition of:```
# def d_state__dt(susceptible: np.ndarray, exposed: np.ndarray, infected: np.ndarray, recovered: np.ndarray, deceased: np.ndarray, social_distancing_intervention_active: np.ndarray, school_closure_intervention_active: np.ndarray, parameters: dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): # (d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt, d_deceased__dt)
# ```
# You can use any parameters you want. Do not define them in the code, instead load them from the parameters dictionary. Here you must model the state differential of the number of people in the country that are susceptible, exposed, infected, recovered, and deceased; with the input actions of social_distancing_intervention_active and school_closure_intervention_active. These input actions are binary, where 0 means the intervention is not active, and 1 means the intervention is active. Give the complete function code.

# Description of the variables:
# * susceptible: Number of people in the country that are susceptible to the virus
# * exposed: Number of people in the country that are exposed to the virus
# * infected: Number of people in the country that are infected with the virus
# * recovered: Number of people in the country that are recovered from the virus
# * deceased: Number of people in the country that are deceased from the virus

# The time units is in days.

# Additionally these variables have the ranges of:
# * susceptible: [{low_range[0]}, {high_range[0]}]
# * exposed: [{low_range[1]}, {high_range[1]}]
# * infected: [{low_range[2]}, {high_range[2]}]
# * recovered: [{low_range[3]}, {high_range[3]}]
# * deceased: [{low_range[4]}, {high_range[4]}]

# Useful to know:
# * The parameters will be optimized to an observed training dataset with the given simulator.
# * It is preferable to decompose the system into differential equations (compartments) if possible.
# * Make the model interpretable and fit the dataset as accurately as possible.
# * Use Python, and only numpy as an external library, which is instantiated already as `np`.
# * You can use any numpy unary functions, for example np.log, np.exp, np.power etc. 
                                        
# Think step-by-step, and then give the complete full working code.
# """
#         elif self.env.env_name == 'COVID-social-distancing':
#             # SEIRD model
#             low_range = self.env.train_data[0].min((0,1))
#             high_range = self.env.train_data[0].max((0,1))
#             initial_prompt = f"""
# You will get a environment task description to code a differential equation simulator for.


# Environment Task description:```
# A COVID-19 model for a country including interventions.
# ```
# You will complete the following function definition of:```
# def d_state__dt(susceptible: np.ndarray, exposed: np.ndarray, infected: np.ndarray, recovered: np.ndarray, deceased: np.ndarray, social_distancing_intervention_active: np.ndarray, parameters: dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): # (d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt, d_deceased__dt)
# ```
# You can use any parameters you want. Do not define them in the code, instead load them from the parameters dictionary. Here you must model the state differential of the number of people in the country that are susceptible, exposed, infected, recovered, and deceased; with the input actions of social_distancing_intervention_active. The input action is binary, where 0 means the intervention is not active, and 1 means the intervention is active. Give the complete function code.

# Description of the variables:
# * susceptible: Number of people in the country that are susceptible to the virus
# * exposed: Number of people in the country that are exposed to the virus
# * infected: Number of people in the country that are infected with the virus
# * recovered: Number of people in the country that are recovered from the virus
# * deceased: Number of people in the country that are deceased from the virus

# The time units is in days.

# Additionally these variables have the ranges of:
# * susceptible: [{low_range[0]}, {high_range[0]}]
# * exposed: [{low_range[1]}, {high_range[1]}]
# * infected: [{low_range[2]}, {high_range[2]}]
# * recovered: [{low_range[3]}, {high_range[3]}]
# * deceased: [{low_range[4]}, {high_range[4]}]

# Useful to know:
# * The parameters will be optimized to an observed training dataset with the given simulator.
# * It is preferable to decompose the system into differential equations (compartments) if possible.
# * Make the model interpretable and fit the dataset as accurately as possible.
# * Use Python, and only numpy as an external library, which is instantiated already as `np`.
# * You can use any numpy unary functions, for example np.log, np.exp, np.power etc. 
                                        
# Think step-by-step, and then give the complete full working code.
# """
#         elif self.env.env_name == 'COVID':
#             # SEIRD model
#             low_range = self.env.train_data[0].min((0,1))
#             high_range = self.env.train_data[0].max((0,1))
#             initial_prompt = f"""
# You will get a environment task description to code a differential equation simulator for.


# Environment Task description:```
# A COVID-19 model for a country.
# ```
# You will complete the following function definition of:```
# def d_state__dt(susceptible: np.ndarray, exposed: np.ndarray, infected: np.ndarray, recovered: np.ndarray, total_population: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray): # (d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt)
# ```
# Here you must model the state differential of the number of people in the country that are susceptible, exposed, infected and recovered. Give the complete function code.

# Description of the variables:
# * susceptible: Number of people in the country that are susceptible to the virus
# * exposed: Number of people in the country that are exposed to the virus
# * infected: Number of people in the country that are infected with the virus
# * recovered: Number of people in the country that are recovered from the virus (including deceased)
# * total_population: Total population of the country, a constant.

# The time units is in days.

# Additionally these variables have the ranges of:
# * susceptible: [{low_range[0]}, {high_range[0]}]
# * exposed: [{low_range[1]}, {high_range[1]}]
# * infected: [{low_range[2]}, {high_range[2]}]
# * recovered: [{low_range[3]}, {high_range[3]}]
# * deceased: [{low_range[4]}, {high_range[4]}]
# * total_population: [10,0000, 10,0000]

# Useful to know:
# * The parameters will be optimized to an observed training dataset with the given simulator.
# * It is preferable to decompose the system into differential equations (compartments) if possible.
# * Make the model interpretable and fit the dataset as accurately as possible.
# * Use Python, and only numpy as an external library, which is instantiated already as `np`.
# * You can use any numpy unary functions, for example np.log, np.exp, np.power etc. 
                                        
# Think step-by-step, and then give the complete full working code.
# """
    