# Business Problem Statement

## Business Context

In last-mile logistics, roughly 10% of all shipments encounter delivery exceptions — failed attempts, address mismatches, damaged packages, refused deliveries, and weather delays. Each failed delivery costs in direct reattempt expenses, while the downstream impact is far steeper: consumers stop shopping with a retailer after just 2–3 failed deliveries, and supply chain disruptions cost companies millions per year.

Despite these costs, most organizations still handle exception triage and resolution manually. Operations staff read through status logs and messy driver notes, cross-reference customer profiles and internal playbooks, decide on a resolution action, and draft customer notifications — all under time pressure. Teams rely on support tickets, tribal knowledge, and rigid rule engines that cannot handle the nuanced, multi-factor judgment calls that real exceptions demand. A VIP customer with a perishable package and a broken gate code on a second attempt requires a fundamentally different resolution than a standard customer's first failed attempt on a non-perishable item — but manual processes treat them with the same slow, inconsistent workflow.

**Key KPIs affected include:**

- Exception resolution time (constrained by manual triage throughput)
- Escalation accuracy (missed or unnecessary supervisor escalations)
- Customer communication quality and personalization
- Cost per exception (reattempt costs, spoilage write-offs, unnecessary truck rolls)
- Customer retention and lifetime value

If unaddressed, logistics providers face rising exception-handling costs, inconsistent service quality, preventable customer churn, and loss of competitive ground to players who are already investing in AI-powered exception detection and response automation.

By applying an AI-powered multi-agent system that ingests delivery logs, cross-references customer profiles and locker availability, retrieves resolution policy from an operational playbook, and generates validated decisions with personalized customer communications, logistics providers can automate the full exception-handling pipeline from detection through resolution, with built-in quality validation and supervisor escalation where policy requires it.

## Objective

The objective is to build a POC of an AI-powered multi-agent delivery exception handling system for the last-mile delivery operations of a mid-sized retailer that:

1. Uses the operator's available data to process exceptions end-to-end.
2. Ingests raw delivery status logs, including noisy, duplicated, and multi-row shipment events, and correctly identifies actionable exceptions versus routine operational noise.
3. Decides the appropriate resolution action (reschedule, reroute to locker, replace, or return to sender) by reasoning over playbook rules, customer context, package constraints, and locker eligibility, while respecting operational policies around perishable handling, fragile thresholds, and locker capacity.
4. Escalates to a human supervisor when policy requires it and avoids unnecessary escalations that waste supervisor capacity.
5. Generates personalized customer notifications with tone, channel, and content calibrated to customer tier and exception severity.
6. Produces auditable decisions with step-by-step rationale, critical validation traces, and system evaluation metrics across five dimensions:
   - Task completion rate
   - Escalation accuracy
   - Tool call accuracy
   - Reasoning trajectory coherence
   - End-to-end latency

The end goal is to demonstrate measurable accuracy and consistency on curated exception scenarios sufficient to justify scaling the approach to real-time operations with live delivery feeds, broader exception taxonomies, and multi-region deployment.

## Data Description

The data for this project comprises three types of files:

| Source | Contents |
| --- | --- |
| SQLite database | Customer profiles and locker locations |
| CSV files | Delivery status logs with free-text driver notes across 10 shipments, and the ground truth labels |
| PDF file | Resolution playbook (exception handling procedures, escalation matrix, locker reroute eligibility rules, customer communication guidelines) |

The details of the PDF file, database tables, CSV files, and their corresponding attributes are provided in the learner notebooks.

## Important Points to Note

**Task completion rate** should be checked using three sub-criteria:

1. Whether the exception classification matches ground truth
2. Whether the resolution action is correct
3. Whether the communication tone is appropriate

A task is complete **only** when all three sub-criteria pass.

**Escalation accuracy** should be computed by matching the system's escalation flag against the expected label in the ground truth.

## Submission Guidelines

There are two ways to work on this project.

| | Full-code | Low-code |
| --- | --- | --- |
| **Who should choose** | Learners who aspire to be in hands-on coding roles in the future, with a focus on building solution code from scratch | Learners who aspire to be in managerial roles in the future — focus on solution review, interpretation, recommendations, and communicating with the business |
| **What is the same** | Perform exploratory data analysis to identify insights and recommendations for the problem | Perform exploratory data analysis to identify insights and recommendations for the problem |
| **What is different** | Focus on code writing: 10–20% grading on the quality of the final code submitted | Focus on business presentation: 10–20% grading on the quality of the final business presentation submitted |
| **Final submission file** | Solution Python notebook from the full-code template submitted in `.html` format | Business presentation in `.pdf` format with problem definition, insights, and recommendations |
| **Submission format** | `.html` | `.pdf` |

The primary purpose of providing these two options is to allow learners to opt for the approach that aligns with their learning aspirations and outcomes.

If you submit a presentation, **only the presentation will be evaluated**. Make sure that all the sections mentioned in the rubric have been covered in your submission.

### Full-code version

1. Download the full-code version of the learner notebook.
2. Follow the instructions provided in the notebook to complete the project.
3. Write down insights and recommendations for the business problems in the comments.
4. Submit only the solution notebook prepared from the learner notebook (format: `.html`).

### Low-code version

1. Download the low-code version of the learner notebook.
2. Follow the instructions provided in the notebook to complete the project.
3. Prepare a business presentation with insights and recommendations for the business problem.
4. Submit only the presentation (format: `.pdf`).

### Submission rules

1. Any assignment found copied or plagiarized from other submissions will not be graded and will be awarded zero marks.
2. Please ensure timely submission. Any submission after the deadline will not be accepted for evaluation.
3. A submission will not be evaluated if it is submitted after the deadline, or if more than one file is submitted.

## Best Practices

### Full-code submissions

- The final Python notebook should be well-documented, with inline comments explaining the functionality of the code and markdown cells containing comments on the observations and insights.
- The notebook should be run from start to finish sequentially before submission.
- Remove all warnings and errors before submission.
- The notebook should be submitted as an HTML file (`.html`), not as a notebook file (`.ipynb`).
- Refer to the FAQ page for common project-related queries.

### Low-code submissions

- The presentation should be made with the Data Science lead of a company as the audience.
- Cover these key points:
  - Business overview of the problem and solution approach
  - Key findings and insights that can drive business decisions
  - Business recommendations
- Explain the key takeaways in an easy-to-understand manner.
- Include the potential benefits of implementing the solution.
- Avoid copying and pasting from the notebook. Show code only when it is the focal point of the presentation.
- The presentation should be submitted as a PDF file (`.pdf`), not as a `.pptx` file.
- Refer to the FAQ page for common project-related queries.

## Rubric

| Criteria | Requirements | Points |
| --- | --- | --- |
| **Data Loading and Model Initialization** | Load the SQLite DB. Load the CSV files. Ingest the PDF for RAG. Set up the LLM and LangSmith traceability. | 5 |
| **Defining Tools** | Define a list of tools for the AI agent: a tool to read delivery logs, a tool to read customer data from the database, a tool to read locker data from the database, and a tool to retrieve playbook excerpts via RAG. The set of tools can be modified as required, as long as they provide the relevant information for the multi-agent system to pass the test cases. | 9 |
| **Defining the Multi-Agent System Architecture** | Define the multi-agent architecture: multi-agent state, sub-agents, prompts, agent nodes, routing, escalation logic, data and agent security, and a visualization of the workflow. The set of sub-agents, nodes, and workflow architecture can be modified as required, as long as they provide the relevant information for the multi-agent system to pass the test cases. | 17 |
| **Test Cases Execution, Evaluation, and Observability** | Execute all test cases. Print the evaluation metrics for each test case. Print traces and document citations for all test cases. Provide observations based on the outputs received for each test case. Compute aggregate metrics combining all test cases. Metrics to compute: task completion rate, escalation accuracy, tool call accuracy, reasoning trajectory coherence, and latency. | 17 |
| **Conclusions and Business Recommendations** | Key takeaways for the business. | 4 |
| **Presentation / Notebook — Overall Quality** | **Presentation:** structure and flow, crispness, visual appeal, conclusion and business recommendations. **Notebook:** structure and flow, well-commented code, conclusion and business recommendations. | 8 |
