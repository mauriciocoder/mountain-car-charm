# Mountain Car Charmed Application

## Overview
This application, designed to run on Juju, is a charmed model trainer and simulator for the Mountain Car problem, with various parameter configurations. Each training session's configuration and simulation results are stored in a MySQL database across two main tables: `training_definition` and `training_result`.

## Key Features
- **Parameterized Training**: Configure model parameters like learning rate (`alpha`), discount factor (`gamma`), exploration rate (`epsilon`), and polynomial dimensions.
- **Simulation and Persistence**: Run simulations and persist the mean trajectory sizes for each model configuration.

## Database Structure
### 1. `training_definition` Table
This table records the parameters for each training session. Each unique configuration is stored as a record, assigned a `training_definition_id`. Fields include:
- **unit_id**: Identifier for the Juju unit running the simulation.
- **feature_type**: Type of feature to use in the model.
- **alpha**: Learning rate.
- **gamma**: Discount factor.
- **epsilon**: Exploration rate.
- **polynomial_dimension**: Degree of polynomial features.
- **protos_per_dimension**: Number of prototypes per dimension.
- **training_sessions**: Number of training iterations.
- **simulations**: Number of simulations per training configuration.

### 2. `training_result` Table
This table stores the results of each training session. Each record is associated with a specific `training_definition_id`, linking it to the model configuration that produced it. Fields include:
- **training_definition_id**: Links to the associated training configuration in the `training_definition` table.
- **trajectory_sizes_mean**: The list of mean trajectory sizes obtained from the simulations, representing the effectiveness of the model under the specified configuration.

## Actions
### `simulation` Action
The `simulation` action trains and simulates models based on a combination of parameters. The results are then saved in the MySQL tables. This action runs in the background to allow other tasks to continue while the training is in progress.

#### Example Command
To initiate the simulation action with a specific set of parameters, use the following command:
```bash
juju run mountain-car-charm/* simulation feature-type=NRB alpha-list='["0.025"]' \
gamma-list='["0.95"]' epsilon-list='["0.5"]' polynomial-dimension-list='["1"]' \
protos-per-dimension-list='["8"]' training-sessions=10 simulations=100 --background
```

In this example:
- **feature-type**: Specifies the feature type (`NRB` in this case).
- **alpha-list**: List of learning rates to iterate over.
- **gamma-list**: List of discount factors.
- **epsilon-list**: List of exploration rates.
- **polynomial-dimension-list**: List of polynomial degrees for feature engineering.
- **protos-per-dimension-list**: Number of prototypes per dimension.
- **training-sessions**: Total training sessions for each configuration.
- **simulations**: Number of simulations to run per configuration.

Each combination of parameters is processed and the trajectory sizes are stored in the `training_result` table, linked to the appropriate configuration in `training_definition`.

### `getdbsecret` Action
The `getdbsecret` action retrieves the database configuration secret, which is a dictionary containing the following fields:
- **dbname**: Name of the database.
- **dbhost**: Hostname of the database server.
- **dbuser**: Username for the database.
- **dbpassword**: Password for the database.
- **dbport**: Port used to connect to the database.

If the secret is unavailable, this action will fail with the message `"Config file not found"`.

#### Example Command
To retrieve the database configuration secret, use the following command:
```bash
juju run-action mountain-car-charm/* getdbsecret
```

## Requirements
- **Juju**: Ensure Juju is installed and the Mountain Car charm is deployed.
- **MySQL Database**: The MySQL database should be accessible to store configuration and simulation results.

To integrate both charms, run the following command:
```bash
juju integrate mountain-car-charm mysql
```
You are now ready to run the simulation action.