#!/usr/bin/env python3
import pymysql
import logging
from ops import ActiveStatus, Framework, CharmBase, ActionEvent, StartEvent, MaintenanceStatus, RelationEvent, main
from ml.mdp_semi_gradient_mountain_car_td_0 import simulate
logger = logging.getLogger(__name__)


class MountainCarCharm(CharmBase):
    def __init__(self, framework: Framework):
        super().__init__(framework)
        framework.observe(self.on.start, self._on_start)
        framework.observe(self.on.database_relation_changed, self._on_database_relation_changed)
        framework.observe(self.on.simulation_action, self._on_simulation_action)
        framework.observe(self.on.getdbsecret_action, self._on_getdbsecret_action)

    def _on_start(self, event: StartEvent):
        """Handle start event."""
        self.unit.status = ActiveStatus("Mountain-Car charm started!")

    def _on_getdbsecret_action(self, event: ActionEvent):
        logger.info("Running _on_getdbsecret_action")
        secret = self.model.get_secret(label="db-secret")
        if secret:
            logger.info(f"secret.content: {secret.get_content()}")
            event.set_results({"config": secret.get_content(), "version": "latest"})
        else:
            event.fail("Config file not found")

    def _on_simulation_action(self, event: ActionEvent):
        self.unit.status = ActiveStatus("Running simulation...")
        logger.info("Running _on_simulation_action")
        """Handle the grant-admin-role action."""
        # Fetch the user parameter from the ActionEvent params dict
        feature_type = event.params["feature-type"]
        alpha_list = list(map(float, event.params["alpha-list"]))
        gamma_list = list(map(float, event.params["gamma-list"]))
        epsilon_list = list(map(float, event.params["epsilon-list"]))
        polynomial_dimension_list = list(map(int, event.params["polynomial-dimension-list"]))
        protos_per_dimension_list = list(map(int, event.params["protos-per-dimension-list"]))
        training_sessions = int(event.params["training-sessions"])
        simulations = int(event.params["simulations"])
        secret = self.model.get_secret(label="db-secret")
        logger.info(f"secret.content: {secret.get_content()}")
        simulate(secret.get_content(), self.unit.name, feature_type, alpha_list, gamma_list, epsilon_list, polynomial_dimension_list, protos_per_dimension_list, training_sessions, simulations)
        event.set_results({
            "success": True
        })
        self.unit.status = ActiveStatus("Simulation action executed")

    def _on_database_relation_changed(self, event: RelationEvent):
        logger.info("Custom log: _on_database_relation_changed")
        self.unit.status = MaintenanceStatus("Setting up database")
        if self.unit.is_leader():
            logger.info("Setting up database for leader")
            try:
                relation_data = event.relation.data.get(event.unit)
                if relation_data:
                    logger.info(f"relation_data: {relation_data}")
                    db_secret = {
                        "dbname": relation_data.get("database"),
                        "dbhost": relation_data.get("host"),
                        "dbuser": relation_data.get("user"),
                        "dbpassword": relation_data.get("password"),
                        "dbport": relation_data.get("port")
                    }
                    logger.info(f"db_secret: {db_secret}")
                    # Adding secret to app so other units can retrieve it
                    self.app.add_secret(content=db_secret, label="db-secret")
                    # Connect to the MySQL database using pymysql
                    connection = pymysql.connect(
                        host=db_secret["dbhost"],
                        user=db_secret["dbuser"],
                        password=db_secret["dbpassword"],
                        database=db_secret["dbname"],
                        port=int(db_secret["dbport"]),
                        cursorclass=pymysql.cursors.DictCursor
                    )
                    # Create the table if it does not exist
                    with connection.cursor() as cursor:
                        create_training_session = """
                        CREATE TABLE IF NOT EXISTS training_definition (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            unit_id VARCHAR(255) NOT NULL,
                            feature_type VARCHAR(255) NOT NULL,
                            alpha FLOAT NOT NULL,
                            gamma FLOAT NOT NULL,
                            epsilon FLOAT NOT NULL,
                            polynomial_dimension INT DEFAULT NULL,
                            protos_per_dimension INT DEFAULT NULL,
                            training_sessions INT NOT NULL,
                            simulations INT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                        cursor.execute(create_training_session)
                        create_training_result = """
                        CREATE TABLE IF NOT EXISTS training_result (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            training_definition_id INT NOT NULL,
                            trajectory_size INT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (training_definition_id) REFERENCES training_definition(id)
                        );
                        """
                        cursor.execute(create_training_result)
                        connection.commit()
                        logger.info("Table training_definition and training_result created or already exists.")
                    connection.close()
            except Exception as e:
                logger.error(f"Failed to setup database: {e}")
                self.unit.status = MaintenanceStatus("Error during database setup")
        else:
            logger.info("Not leader, skipping database setup")
            # Example of how to retrieve the secret set in another unit
            secret = self.model.get_secret(label="db-secret")
            logger.info(f"secret.content: {secret.get_content()}")
        self.unit.status = ActiveStatus("Database setup completed")

if __name__ == "__main__":  # pragma: nocover
    main(MountainCarCharm)  # type: ignore
