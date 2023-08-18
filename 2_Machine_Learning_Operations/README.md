# Operationalizing Machine Learning

In this project, we will continue working with the Bank Marketing dataset. We will utilize Azure to set up a cloud-based machine learning production model, carry out its deployment, and access it as consumers. Additionally, we will construct, release, and use a pipeline.

## Architectural Diagram

<img src="./screenshots/project_architecture.png" alt="project_architecture" width="800"/>

## Key Steps

In the initial step, we establish authentication. Next, we perform an Automated ML Experiment. Once completed, we deploy the most optimal model and enable logging. Subsequently, we generate Swagger Documentation and proceed to consume model endpoints. We then create and publish a pipeline, ensuring seamless workflow. Lastly, comprehensive documentation is prepared to encompass the entire process.

## Screen Recording

The link to the screen recording of the project: https://youtu.be/n904NpzQUE4

> It was really funny when the Udacity Virtual Machine just died at the end of the video, at the "best" time. Lucky me! :smile:

The screenshots can be found under the `./screenshots` folder:
1. [Registered Dataset](./screenshots/registered_dataset.png)
1. [Experiment Completed](./screenshots/experiment_completed.png)
1. [Best Model](./screenshots/best_model.png)
1. [Application Insights Enabled](./screenshots/application_insights_enabled.png)
1. [Logs Output](./screenshots/logs_output.png)
1. [SwaggerUI Bad Port](./screenshots/swagger_browser_port_80.png)
1. [SwaggerUI Good Port](./screenshots/swagger_browser_port_9000.png)
1. [SwaggerUI Terminal](./screenshots/swagger_terminal.png)
1. [Endpoint Output](./screenshots/endpoint_output.png)
1. [Apache Benchmark Output](./screenshots/apache_benchmark.png)


## Tips and Tricks

- Compute cluster
    - At the second step, the compute cluster name is typod. Use the `STANDARD_D2_V2` virtual machine instead of `STANDARD_D12_V2` because you will not have permission to create it.
    - Set the minimum node to 1 and change priority from **Dedicated** to **Low Priority** which saves compute cost.
- Model deployment
    - Navigate to **Jobs** > **Model** tab > press **Deploy** button
    - Select the compute type as `Azure Container Instance` and check in the **Enable Logging** and the **Enable Applications Insights** under the advanced settings.
- Python and Bash scripts
    - Execute the *Python scripts* with the **Windows Command Prompt** because it will freeze if you use Git Bash.
    - Execute the *Bash scripts* with the **Git Bash** because Windows Command Prompt was no WSL therefore the `bash` command will not work.
- Endpoint script
    - Retrieve the authentication information from the **Endpoints** > **Consume** tab.
    - Copy and assign the **REST endpoint** value to the `scoring_uri` variable.
    - Copy and assign the one of the **Authentication** key to the `key` variable.
    - Make sure that the `data` dictionary match with the example data that is provided under the **Consume** tab. Otherwise, you will got error in the response.
- Swagger UI
    - First, download the `swagger.json` under the **Endpoints** > **Details** tab > **Swagger URI** section. In the Udacity VM, both `wget` and `curl` did not work for me so create an empty `swagger.json` file and open the URI and copy the content to the JSON file.
    - You need to change the port from 80 to other port that is more than 8000 (e.g. 9000) otherwise the Swagger UI webpage will only show "It works" text. Go to the `swagger/swagger.sh` script and change the port at the `-p` option.
    - Execute both the `serve.py` and the `swagger.sh` script before opening the SwaggerUI (<http://localhost>) webpage. Then update the explore URL to <http://localhost:8000/swagger.json> and click on the **Explore** button.
- Apache Benchmark
    - Similarly to the endpoint script, get the authentication information.
    - Replace the `REPLACE_WITH_KEY` with the `Primary Key` or with the `Secondary Key` authentication key.
    - Replace the `http://REPLACE_WITH_API_URL/score` with the **REST endpoint** URL value.