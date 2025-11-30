# Design Notes
By Elizabeth Bradley
### Comprehensive Explanation of Multi-Agent System
https://github.com/Jaroken/AgenticAI_Udacity/tree/main/Multi_Agent_AgenticAI_Nano_Course4_project
In this section I will give a complete overview of the multi-agent system I developed for the fictitious Beaver’s Choice Paper Company. More specifically, I will explain the agent workflow diagram, the roles of the agents in the system, and the decisions made that led to the final agentic system architecture.

#### Agent Workflow Diagram Explanation:
In the agentic AI diagram it is shown that the incoming customer request (user prompt) is given to the orchestration agent, who coordinates with the other agents to generate a final response.

In this architecture the orchestration agent has tools available, including functions that call other agents and ones that do not. Some functions call either an inventory, quoting, or ordering agent, that each have their own tools they can call. There is also a data extraction agent, who has one purpose, which is to extract needed information and store it in a shared state.

The orchestration agent is tasked with running the whole show. They can make the following tool calls: record_email_details, generate_financial_report_dict, get_delivery_address, determine_stock_needs, call_inventory_manager, call_quoting_manager, and call_ordering_manager. The tools can be broken down into a few categories.




##### Tools that call another agent for a general function:
This includes the tools: call_inventory_manager, call_quoting_manager, and call_ordering_manager. These tools are for the orchestration agent to call specific agents to assist in tasks related to these agents specialisation. The orchestration agent is given suggestions on which agents should do what in the system prompt, but it is up to the orchestration agent to determine specifically what it wants from these agents.

##### Tools that call an agent to perform a specific function:
Record_email_details rigidly asks a templated request of the data_extraction_agent to parse the user input for important information. The data_extraction_agent does not have any tools available itself and does not receive custom input from the orchestrator.
Determine_stock_needs also rigidly asks the inventory agent to assess the stock needs based on the data in the shared state. Unlike the tool data_extraction_agent, the inventory agent does have the latitude to call its tools if it needs more information that was not provided already in the shared state.

##### Tools that do not call out agents:
The functions get_delivery_address and  generate_financial_report_dict simply execute the functions, but do not call other agents. The orchestration agent runs these tools entirely.

More details on the other tools and roles of the worker agents will be discussed in the next section. 

#### Roles of the Agents:
##### Orchestration 
The orchestration agent, already outlined above, does as the name implies. They orchestrate the whole show. They are given a suggested roadmap on how to complete their task, but they ultimately call the shots on what tools to call to get to the final response.

##### Inventory
The inventory agent manages the stock. They carry out the tasks of reporting what stock is available by querying the database and also determine needed stock to meet client demands. 

They can address specific questions from the orchestrator by calling specific tools available to it, including: get_inventory_for_date (which gives a large report of all stock items) and the more targeted get_stock_level_item which checks stock availability for a specific item.

This agent also performs the important work of calculating stock needs when determine_stock_needs is called by the orchestrator.                                       

##### Quoting
The quoting agent performs the main function of informing the orchestrator on setting prices and determining cash availability. They look up historical quotes and call the database to determine cash availability.
The quoting agent can call specific tools to carry out the tasks asked by the orchestrator agent, including get_cash_balance_value and search_quote_history_retrieve. get_cash_balance returns the cash given a specific date, while quote_history retrieves quotes provided for specific items in the past.

With these tools the quoting agent helps the orchestrator set prices and determine tax amounts.

##### Ordering
The ordering agent’s role is to schedule deliveries and record orders. They calculate delivery times from the supplier to beaver’s choice and from beaver’s choice to the customer to help the orchestration agent determine order viability. They also put in stock orders and record sales in the database.

This agent has the following tools available to it to accomplish its goals: 
get_supplier_delivery_date_estimate: gets ETA to Beaver’s Choice from supplier
get_customer_delivery_date_estimate: get ETA to the customer from Beaver’s Choice. 
create_transaction_record: which enters either a stock order or a customer sale

The responsibility of this agent is quite large given it both determines feasibility and records sales and stock orders for the orchestrator. If more agents were allotted for this project I would suggest splitting its responsibilities.

##### Data Extraction

The data extraction agent has one simple purpose. That is to parse customer responses for the following:  goals_of_request, request_date, items_names_requested_from_customer, items_names_requested_match_from_financial_report, quantity_of_items_requested, desired_delivery_date. These essential points of information are crucial to the overall system and warranted a devoted agent for just this task.

 
#### Decision Making Process That Led to the Chosen Architecture:
 
The chosen framework uses pydantic ai. This framework was chosen as it has been described as the most enterprise level framework among the suggested frameworks to use for this project. I wanted to gain experience in a framework I could adapt for an enterprise level system. 

I decided to use an orchestration style design as I needed a way to coordinate the various prescribed agents (inventory, quoting, and ordering) and I knew that the various agents were dependent on the outputs of other agents and I needed a way to glue all these agents and outputs together into a cohesive output for the user. This was the primary motivator for choosing this architecture.

Which agents had certain tools evolved over time. This evolved to help improve the functionality as I realised through extensive testing and iterating that certain tools were better called directly from the orchestrator or by a worker agent. My initial design changed dramatically over time as my understanding of the inventory, quoting and ordering process of companies improved. I also responded to seeing how certain tools and agents would get stuck in response/tool calling loops and blow through budgeted resources. Even with the improvements I had to modify the response budgets and retries to enable the system to handle the most demanding of the customer queries without erroring out.

To assist with keeping data straight in the agentic system I employed a few techniques. First I chose to heavily utilize pydantic data structures to keep data properly formatted and neat for tool calls. Secondly I used a shared state system and only allowed specific tool calls to write to the shared state in order to keep the information shared well structured and less prone to hallucinations when passing data between agents.

### Evaluation of test_results.csv
In this section I will discuss the results of the test_results.csv and outline the strengths and at least two areas of improvement for the agentic system I developed. 

#### Strengths of the Developed Agentic System
I am extremely happy with the performance of this system overall. Here are some of the reasons why I think this system is robust.
Prompt engineering: Since the scenario the system was tasked with handling was artificial I had to control for some of the ‘smarter’ features inherent in chatgpt. 
The system would default to use today's date as the current date when using tools, which is entirely correct in practice but the project demanded to treat the date requested as the current date.
The system would want to ask for clarification before finishing a transaction. I had to explicitly make it a greedier system, to enter orders for items that were feasible without clarifications and only ask for clarification on infeasible items.
I had to override other very sensible decisions the system agentic system was trying to make. Often it would troubleshoot infeasible delivery times, with solutions such as offering expedited shipping. I had to make up fake addresses to fill in the logical gap of not providing the agentic system with  shipping addresses while also asking for delivery time estimates. 
The system successfully moved the needle and placed orders when I wanted it to and also ordered stock as well. From the examples I reviewed I did see it performed accurately - that is ordering the correct amounts and sharing why it could not complete orders in a customer friendly style.
Structured output and shared state: I think the pydantic data structures and shared state help maintain consistency well
Additional focused tools to help keep scope in check and execute important tasks. I was happy to add in the  determine_stock_needs to formalize that evaluation as opposed to just letting the system figure it out.
I put in some decent logging using logfire and I also saved the output to a txt file for further evaluation. I found this helpful when analysing performance and I will continue to use this approach in the future.

#### Suggestions for further Improvements
Although I am quite happy with the system, there seems to be much that can be done to improve the system with sufficient time and resources (I used up lots of my openai credits testing and iterating). Some improvements I believe could help the system are:
Split the create_transaction_record into two unique tools - one for adding stock orders, which can be used by the inventory agent, and another for sales orders that can be used by the ordering agent. I think this division would make more sense and would require less cross agent communication to execute tasks
Record quotes provided. I believe the system does not currently record the quotes provided to the customer, instead they are just saved for evaluation purposes. In a production system a long term memory solution would be very helpful. This could be done by saving the prompt and output in a json and saving it in a data repository, or to add a record to the existing system that is used to query historical quotes.
I think there is room to parallelize the tasks a little further. I believe I could have divided up the tools in a way that allowed each agent to work independently more and be less dependent on sequential executions. I think this would have sped up the time to completion for the system, which was fairly long.
I also noticed, that despite my best efforts, it kept seeming like nonsensical tool calling took place where the same tools were called for various slightly different values. This seemed to me to be inefficient and if it could be improved would speed up time to completion and save resources.
