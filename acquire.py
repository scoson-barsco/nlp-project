"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import unicodedata
import re
import json
import numpy as np

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
import acquire
from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['Azure/solution-center',
'Azure/azure-docs-sdk-dotnet',
'Azure/azure-rest-api-specs',
'Azure/AgentBaker',
'Azure/bicep',
'Azure/azureml-examples',
'Azure/communication-ui-library',
'Azure/azure-linux-extensions',
'Azure/azureml-assets',
'Azure/aks-app-routing-operator',
'Azure/Community-Policy',
'Azure/Azurite',
'Azure/azure-functions-postgresql-extension',
'Azure/azure-sdk-for-net',
'Azure/azure-openapi-validator',
'Azure/azure-sdk-for-js',
'Azure/AKS',
'Azure/azure-sdk-for-cpp',
'Azure/azure-sdk-for-java',
'Azure/azure-kusto-go',
'Azure/azure-dev',
'Azure/azure-cli',
'Azure/azure-sdk-for-python',
'Azure/azure-cli-extensions',
'Azure/azure-functions-nodejs-worker',
'Azure/iotedge',
'Azure/azure-cosmos-dotnet-v3',
'Azure/AppConfiguration-DotnetProvider',
'Azure/azure-docs-sdk-java',
'Azure/azure-sdk-assets',
'Azure/azure-sdk-for-rust',
'Azure/azure-functions-dotnet-worker',
'Azure/oav',
'Azure/cosmos-explorer',
'Azure/azure-container-networking',
'Azure/BatchExplorer',
'Azure/iot-identity-service',
'Azure/azure-sdk-tools',
'Azure/fleet',
'Azure/azure-osconfig',
'Azure/terraform',
'Azure/cyclecloud-slurm',
'Azure/ARO-RP',
'Azure/azure-functions-host',
'Azure/LogicAppsUX',
'Azure/azure-iot-hub-node',
'Azure/communication-ui-library-ios',
'Azure/azure-sdk',
'Azure/Azure-Sentinel',
'Azure/azure-sdk-for-go',
'Azure/gpt-rag-ingestion',
'Azure/gpt-rag-orchestrator',
'Azure/gpt-rag-frontend',
'Azure/azure-sdk-for-c',
'Azure/azure-cli-dev-tools',
'Azure/azure-docs-powershell-azuread',
'Azure/azure-functions-sql-extension',
'Azure/aks-engine-azurestack',
'Azure/opendigitaltwins-tools',
'Azure/azure-storage-azcopy',
'Azure/azure-sphere-gallery',
'Azure/aks-tls-bootstrap',
'Azure/azure-spring-apps-landing-zone-accelerator',
'Azure/azure-functions-ux',
'Azure/Bridge-To-Kubernetes',
'Azure/azure-functions-docker',
'Azure/aca-landing-zone-accelerator',
'Azure/Wordpress-on-Linux-App-Service-plugins',
'Azure/azure-powershell',
'Azure/api-management-developer-portal',
'Azure/WALinuxAgent',
'Azure/terraform-azurerm-lz-vending',
'Azure/avdaccelerator',
'Azure/azure-uamqp-python',
'Azure/azure-maps-creator-onboarding-tool',
'Azure/windup-rulesets',
'Azure/Azure-Network-Security',
'Azure/SAP-on-Azure-Scripts-and-Utilities',
'Azure/gpt-rag',
'Azure/data-api-builder',
'Azure/Industrial-IoT',
'Azure/load-testing',
'Azure/observable-python-azure-functions',
'Azure/cortana-intelligence-price-analytics',
'Azure/azure-functions-extension-bundles',
'Azure/azure-kusto-java',
'Azure/azure-openai-workshop',
'Azure/libguestfs',
'Azure/azure-storage-fuse',
'Azure/azure-rest-api-specs-examples',
'Azure/actions',
'Azure/SAP-automation-samples',
'Azure/azqr',
'Azure/cognitive-search-vector-pr',
'Azure/ahm-templates',
'Azure/GuardrailsSolutionAccelerator',
'Azure/aaz',
'Azure/reliability-workbook',
'Azure/Azure-AppServices-Diagnostics-Portal',
'Azure/azure-service-operator',
'Azure/awesome-azd',
'Azure/azure-resource-manager-schemas',
'Azure/generator-jhipster-azure-spring-apps',
'Azure/dotnet-extensions-experimental',
'Azure/ALZ-Bicep',
'Azure/RAI-vNext-Preview',
'Azure/Oracle-Workloads-for-Azure',
'Azure/c-logging',
'Azure/MQTTBrokerPrivatePreview',
'Azure/c-pal',
'Azure/azure-functions-core-tools',
'Azure/prometheus-collector',
'Azure/azure-functions-powershell-worker',
'Azure/acr-cli',
'Azure/acr',
'Azure/sonic-buildimage-msft',
'Azure/LinuxPatchExtension',
'Azure/api-management-policy-snippets',
'Azure/azure-functions-python-worker',
'Azure/ahds-reference-architecture',
'Azure/iot-hub-device-update',
'Azure/portaldocs',
'Azure/azure-functions-templates',
'Azure/azure-iot-cli-extension',
'Azure/AppService',
'Azure/secrets-store-csi-driver-provider-azure',
'Azure/Commercial-Marketplace-SaaS-Accelerator',
'Azure/azure-policy',
'Azure/kubernetes-carbon-intensity-exporter',
'Azure/ADX-in-a-Day-Lab2',
'Azure/ADX-in-a-Day-Lab1',
'Azure/terraform-azure-container-apps',
'Azure/enterprise-azure-policy-as-code',
'Azure/azure-functions-openapi-extension',
'Azure/terraform-azurerm-vnet',
'Azure/login',
'Azure/airflow-provider-azure-machinelearning',
'Azure/azure-functions-redis-extension',
'Azure/MDEASM-Solutions',
'Azure/communication-ui-library-android',
'Azure/go-asynctask',
'Azure/azure-iot-sdk-c',
'Azure/azure-functions-python-library',
'Azure/ResourceModules',
'Azure/terratest-terraform-fluent',
'Azure/repair-script-library',
'Azure/iot-plugandplay-models-tools',
'Azure/draft',
'Azure/terraform-azurerm-network',
'Azure/woc-benchmarking',
'Azure/terraform-azurerm-database',
'Azure/nxtools',
'Azure/AppConfiguration',
'Azure/azure-spring-initializr',
'Azure/terraform-azurerm-compute',
'Azure/eraser-scanner-template',
'Azure/terraform-azurerm-loadbalancer',
'Azure/azure-workload-identity',
'Azure/alz-monitor',
'Azure/Avere',
'Azure/appservice-landing-zone-accelerator',
'Azure/business-process-automation',
'Azure/plato',
'Azure/kubernetes-kms',
'Azure/terraform-azurerm-aks',
'Azure/ccf-identity',
'Azure/cadl-ranch',
'Azure/azure-iot-middleware-freertos',
'Azure/azure-signalr',
'Azure/AKS-Construction',
'Azure/webapps-deploy',
'Azure/arm-ttk',
'Azure/meta-iotedge',
'Azure/bicep-registry-modules',
'Azure/terraform-azurerm-subnets',
'Azure/dotnet-template-azure-iot-edge-module',
'Azure/terraform-azure-mdc-defender-plans-azure',
'Azure/azure-webpubsub',
'Azure/azure-cosmosdb-ads-extension',
'Azure/Enterprise-Scale',
'Azure/MS-AMP-Examples',
'Azure/ArcOnAVS',
'Azure/ArcOnAVSInternal',
'Azure/terraform-azapi-hybridcontainerservice',
'Azure/cli',
'Azure/terraform-azurerm-alz-management',
'Azure/terraform-azurerm-openai',
'Azure/terraform-azurerm-postgresql',
'Azure/terraform-azurerm-virtual-machine',
'Azure/azure-functions-kafka-extension',
'Azure/ads-extension-mongo-migration-assets',
'Azure/azure-quickstart-templates',
'Azure/azure-openai-samples',
'Azure/terraform-azurerm-hubnetworking',
'Azure/eraser',
'Azure/azure-powershell-common',
'Azure/terraform-verified-module',
'Azure/terraform-azurerm-network-security-group',
'Azure/terraform-module-test-helper',
'Azure/setup-helm',
'Azure/bicep-types',
'Azure/bicep-types-k8s',
'Azure/MS-AMP',
'Azure/bicep-types-az',
'Azure/azure-sdk-for-ios',
'Azure/azure-sdk-for-android',
'Azure/Microsoft-Defender-for-Cloud',
'Azure/carnegie-mop',
'Azure/mtm-tech-enablement-labs',
'Azure/monaco-kusto',
'Azure/awps-webapp-sample',
'Azure/Azure-DataFactory',
'Azure/mec-app-solution-accelerator',
'Azure/cosmos-js-sdk-debug-tools',
'Azure/aca-review-apps',
'Azure/aks-devx-tools',
'Azure/azure-api-management-devops-resource-kit',
'Azure/ipam',
'Azure/azure-relay',
'Azure/OCPCHINATECH',
'Azure/umock-c',
'Azure/c-testrunnerswitcher',
'Azure/azure-docs-bicep-samples',
'Azure/durabletask',
'Azure/azure-db-benchmarking',
'Azure/azure-functions-nodejs-library',
'Azure/azure-mobile-apps',
'Azure/azure-functions-java-worker',
'Azure/kubernetes-hackfest',
'Azure/azure-functions-durable-extension',
'Azure/IoTC-Reserved-IoTHub-List',
'Azure/missionlz',
'Azure/azure-functions-dapr-extension',
'Azure/api-management-self-hosted-gateway',
'Azure/trusted-internet-connection',
'Azure/Mission-Critical-Online',
'Azure/Azure-Center-for-SAP-solutions-preview',
'Azure/FTALive-Sessions',
'Azure/azure-functions-on-container-apps',
'Azure/azure-sdk-for-php',
'Azure/tfmod-scaffold',
'Azure/ctest',
'Azure/azure-iotedge',
'Azure/generator-azure-iot-edge-module',
'Azure/azapi2azurerm',
'Azure/discover-java-apps',
'Azure/blackbelt-aks-hackfest',
'Azure/video-analyzer-widgets',
'Azure/azure-functions-tooling-feed',
'Azure/cxp-isv',
'Azure/azure-code-signing-action',
'Azure/azurehpc',
'Azure/azure-remote-rendering',
'Azure/linux-image-validations',
'Azure/com-wrapper',
'Azure/terraform-azurerm-caf-enterprise-scale',
'Azure/kubectl-aks',
'Azure/blobxfer',
'Azure/MachineLearningNotebooks',
'Azure/azure-functions-vs-build-sdk',
'Azure/moby-packaging',
'Azure/azure-iot-sdk-csharp',
'Azure/macro-utils-c',
'Azure/Feathr',
'Azure/Enterprise-Scale-for-AVS',
'Azure/container-apps-deploy-action',
'Azure/run-command-extension-linux',
'Azure/azure-automation-go-worker',
'Azure/OpenShift',
'Azure/azurehpc-health-checks',
'Azure/CanadaPubSecALZ',
'Azure/rg-cleanup',
'Azure/Communication',
'Azure/Azure-Maps-Style-Editor',
'Azure/AzureML-Containers',
'Azure/AZNFS-mount',
'Azure/container-service-for-azure-china',
'Azure/azure-docs-powershell-samples',
'Azure/machinelearning-model-optimizer-preview',
'Azure/Azure-Proactive-Resiliency-Library',
'Azure/ALZ-PowerShell-Module',
'Azure/sf-c-util',
'Azure/clds',
'Azure/c-util',
'Azure/caf-terraform-landingzones',
'Azure/jp-techdocs',
'Azure/kubelogin',
'Azure/ADXIoTAnalytics',
'Azure/rhel-jboss-templates',
'Azure/azure-sphere-samples',
'Azure/placement-policy-scheduler-plugins',
'Azure/azure-functions-nodejs-e2e-tests',
'Azure/application-gateway-kubernetes-ingress',
'Azure/azure-iot-sdk-java',
'Azure/azure-iot-service-sdk-java',
'Azure/gen-cv',
'Azure/azhpc-images',
'Azure/SQL-Connectivity-Checker',
'Azure/azure-data-labs-modules',
'Azure/Azure-Governance-Visualizer',
'Azure/Service-Fabric-Troubleshooting-Guides',
'Azure/deployment-environments',
'Azure/aaz-dev-tools',
'Azure/iot-plugandplay-models',
'Azure/vscode-bridge-to-kubernetes',
'Azure/azure-functions-durable-js',
'Azure/go-asyncjob',
'Azure/azure-stream-analytics',
'Azure/InnovationEngine',
'Azure/SQL-Migration-AzureSQL-PoC',
'Azure/intelligent-app-workshop',
'Azure/AKS-DevSecOps-Workshop',
'Azure/partnercenter-cli-extension',
'Azure/deployment-stacks',
'Azure/AzureGovernedPipelines',
'Azure/image-rootfs-scanner',
'Azure/acr-build',
'Azure/azure-notificationhubs-android',
'Azure/azure-kusto-spark',
'Azure/aks-periscope',
'Azure/functions-action',
'Azure/AzureMonitorForVMs-ArmTemplates',
'Azure/CacheBrowns',
'Azure/Azure-Orbital-STAC',
'Azure/ip-masq-agent-v2',
'Azure/aad-pod-identity',
'Azure/go-armbalancer',
'Azure/run-command-handler-linux',
'Azure/sap-automation',
'Azure/azure-saas',
'Azure/azure-cosmos-db-query-editor',
'Azure/spring-apps-deploy',
'Azure/kubernetes-volume-drivers',
'Azure/azapi-vscode',
'Azure/terraform-provider-azapi',
'Azure/azurecosmosdb-vercel-starter',
'Azure/azurestack-powershell',
'Azure/hpcpack',
'Azure/azapi-lsp',
'Azure/static-web-apps-cli',
'Azure/AKS-Edge',
'Azure/aztfexport',
'Azure/azure-functions-durable-powershell',
'Azure/AzLoadBalancerMigration',
'Azure/data-product-analytics',
'Azure/data-product-streaming',
'Azure/data-product-batch',
'Azure/data-landing-zone',
'Azure/confidential-computing-cvm-guest-attestation',
'Azure/YCSB',
'Azure/azure-databricks-client',
'Azure/Moneo',
'Azure/azure-functions-durable-python',
'Azure/azure-sdk-for-c-arduino',
'Azure/IoTTrainingPack',
'Azure/optimized-pytorch-on-databricks-and-fabric',
'Azure/cyclecloud-symphony',
'Azure/azure-kusto-python',
'Azure/Mission-Critical-Connected',
'Azure/azure-kusto-node',
'Azure/terraform-azure-modules',
'Azure/azure-arc-validation',
'Azure/azure-iac-workshop-content',
'Azure/azure-cosmosdb-spark',
'Azure/azure-cosmosdb-java',
'Azure/azure-acr-plugin',
'Azure/grpc-go-redact',
'Azure/Azure-Maps-plugin-for-QGIS',
'Azure/c-build-tools',
'Azure/homebrew-azd',
'Azure/azure-iot-sdk-node',
'Azure/Project-Cerberus',
'Azure/template-analyzer',
'Azure/azure-iot-sdk-python',
'Azure/k8s-deploy',
'Azure/azure-storage-av-automation',
'Azure/azure-kusto-splunk',
'Azure/acr-builder',
'Azure/app-service-windows-containers',
'Azure/azure-uamqp-c',
'Azure/iotedge-lorawan-starterkit',
'Azure/api-center-preview',
'Azure/azure-utpm-c',
'Azure/azure-uhttp-c',
'Azure/azure-umqtt-c',
'Azure/static-web-apps',
'Azure/azure-c-shared-utility',
'Azure/app-service-linux-docs',
'Azure/ace-luna',
'Azure/bicep-lz-vending',
'Azure/azure-maps-ios-sdk-distribution',
'Azure/aca-dotnet-workshop',
'Azure/terraform-azurerm-naming',
'Azure/azure-spring-suitability-rules',
'Azure/logstash-output-kusto',
'Azure/iot-sdks-e2e-fx',
'Azure/Azure-Sentinel-Notebooks',
'Azure/reddog-solutions',
'Azure/mlops-v2-cv-demo',
'Azure/mlops-project-template',
'Azure/avocado',
'Azure/RDS-Templates',
'Azure/KDAHackathon',
'Azure/acr-task-commands',
'Azure/MDTI-Solutions',
'Azure/iot-edge-config',
'Azure/azure-data-labs',
'Azure/reliable-web-app-pattern-dotnet-workshop',
'Azure/aks-traffic-manager',
'Azure/azure-kusto-log4j',
'Azure/azure-notificationhubs-dotnet',
'Azure/azure-spring-rewrite',
'Azure/EasyAuthForK8s',
'Azure/bicep-shared-tools',
'Azure/azure-devops-cli-extension',
'Azure/azure-kusto-nlog-sink',
'Azure/azure-blob-storage-file-upload-utility',
'Azure/azure-webjobs-sdk-extensions',
'Azure/reliable-web-app-pattern-dotnet',
'Azure/aspnet-redis-providers',
'Azure/logicapps',
'Azure/azure-netapp-files',
'Azure/microsoft-graph-docs',
'Azure/Health-Data-and-AI-Blueprint',
'Azure/Integration-Services-Landing-Zone-Accelerator',
'Azure/azureml-oss-models',
'Azure/AKS-Landing-Zone-Accelerator',
'Azure/Microsoft365R',
'Azure/azure-signalr-bench',
'Azure/openai-at-scale',
'Azure/azure-relay-dotnet',
'Azure/go-amqp',
'Azure/data-management-zone',
'Azure/azure-relay-java',
'Azure/ShieldGuard',
'Azure/azure-webjobs-sdk',
'Azure/iot-central-paad',
'Azure/HPC-Accelerator',
'Azure/azure-postgresql',
'Azure/CloudShell',
'Azure/hbase-utils',
'Azure/Moodle',
'Azure/azure-capi-cli-extension',
'Azure/azure-cli-1',
'Azure/Performance-Efficiency-Scripts-SAP-ORA',
'Azure/homebrew-kubelogin',
'Azure/reddog-aks-workshop',
'Azure/autorest',
'Azure/sap-automation-bootstrap',
'Azure/fta-japan',
'Azure/notation-azure-kv',
'Azure/serilog-sinks-azuredataexplorer',
'Azure/mlops-starter-sklearn',
'Azure/azure-iot-explorer',
'Azure/azure-kusto-webexplorer-embedding',
'Azure/Storage',
'Azure/Azure-Data-Factory-Integration-Runtime-in-Windows-Container',
'Azure/openai-samples',
'Azure/KeyVault-AccessPolicyToRBAC-CompareTool',
'Azure/redact',
'Azure/azure-kusto-dotnet',
'Azure/DaaS',
'Azure/azvmimagebuilder',
'Azure/azure-functions-powershell-library',
'Azure/azure-event-hubs-for-kafka',
'Azure/aks-gpu',
'Azure/azure-vm-utils',
'Azure/azure-api-style-guide',
'Azure/GPT_ALE',
'Azure/feast-azure',
'Azure/azure-kusto-labs',
'Azure/azure-functions-devops-build',
'Azure/buffalo-azure',
'Azure/azure-keyvault-cli-extension',
'Azure/azure-sdk-vcpkg-betas',
'Azure/azure-notificationhubs-xamarin',
'Azure/osdu-data-load-tno',
'Azure/kafka-private-preview',
'Azure/diagnostics-eventflow',
'Azure/azure-powershell-migration',
'Azure/azure-javaee-iaas',
'Azure/azure-sdk-actions',
'Azure/bicep-registry-providers',
'Azure/CommercialConfidentialCompute',
'Azure/CCOInsights',
'Azure/hpc-cache',
'Azure/decisionAI-plugin',
'Azure/Azure-Governance-Visualizer-Accelerator',
'Azure/azure-multiapi-storage-python',
'Azure/terraform-azurerm-resource-group',
'Azure/terraform-azurerm-search-service',
'Azure/terraform-test',
'Azure/azure_preview_modules',
'Azure/api-management-samples',
'Azure/Azure-kusto-opentelemetry-demo',
'Azure/Commercial-Marketplace-SaaS-Accelerator-Offer',
'Azure/azure-signalr-test',
'Azure/embedded-wireless-framework',
'Azure/aks-baseline-windows',
'Azure/fta-identity',
'Azure/azure-relay-aspnetserver',
'Azure/active-directory-dotnet-graphapi-b2bportal-web',
'Azure/azure-cosmos-dotnet-v2',
'Azure/fta-internalbusinessapps',
'Azure/azure-libraries-for-net',
'Azure/openapi-markdown',
'Azure/azure-iot-sdks',
'Azure/azure-cosmosdb-js-server',
'Azure/blobporter',
'Azure/appservice-zipped-templates',
'Azure/boilerplate-azurefunctions',
'Azure/oms-agent-for-linux-boshrelease',
'Azure/acr-docker-credential-helper',
'Azure/app-service-announcements-discussions',
'Azure/coco-framework',
'Azure/azure-pixel-tracker-arm',
'Azure/ms-rest-azure-js',
'Azure/ms-rest-azure-env',
'Azure/azure-pixel-tracker',
'Azure/DevOps-For-AI-Apps',
'Azure/autorest-clientruntime-for-swift',
'Azure/fedramp-iaas-webapp',
'Azure/azure-storage-queue-go',
'Azure/Liftnshiftplus',
'Azure/hpcpack-template-2012r2',
'Azure/MachineLearningSamples-NotebookTemplate',
'Azure/smarthotels360-azure',
'Azure/cloud-debug-tools',
'Azure/autorest-extension-base',
'Azure/jenkins',
'Azure/terraform-azurerm-encryptedmanageddisk',
'Azure/mirrorcat',
'Azure/terramodtest',
'Azure/autorest-extension-helloworld',
'Azure/GBB-SEC',
'Azure/azure-storage-queue-php',
'Azure/azure-functions-docker-python-sample',
'Azure/utility_functions_in_ROC_space',
'Azure/maven-bundler',
'Azure/azure-cosmos-js',
'Azure/CosmosDB-GBB-Hackathon',
'Azure/aml-real-time-ai',
'Azure/AML-AirField',
'Azure/CIQS-Azure-Cli-Extension',
'Azure/git2grid',
'Azure/hpcpack-acm-api-python',
'Azure/hpcpack-acm-cli',
'Azure/cyclecloud-nfs',
'Azure/cyclecloud-docker',
'Azure/cyclecloud-ubercloud',
'Azure/cyclecloud-conda',
'Azure/cyclecloud-tractor',
'Azure/azure-qube',
'Azure/cyclecloud-container',
'Azure/azure-storage-cpplite',
'Azure/terraform-azurerm-routetable',
'Azure/azure-cef-sdk',
'Azure/ignite-aks-bestpractices',
'Azure/WebAndRazorWithVisualStudioForMac',
'Azure/Ansible',
'Azure/Fleet-PRSE',
'Azure/DigitalTwins-Helper-Library',
'Azure/terraform-azurerm-disk-snapshot',
'Azure/dotnet-template-azure-iot-edge-module-visual-studio',
'Azure/terraform-azurerm-vm-extension-msi',
'Azure/terraform-azurerm-storage-account',
'Azure/cyclecloud-nvidia-gpu-cloud',
'Azure/azure-kusto-samples-dotnet',
'Azure/azure-tokens',
'Azure/terraform-azurerm-appgw-ingress-k8s-cluster',
'Azure/spark-cdm',
'Azure/hpcpack-acm-api-dotnet',
'Azure/AMLPipelines',
'Azure/hpcpack-acm-ps',
'Azure/azure-functions-performance-scenarios',
'Azure/SwiftPM-AzureCommunicationCommon',
'Azure/azure-notificationhubs-samples',
'Azure/azure-remote-rendering-asset-tool',
'Azure/avslabs',
'Azure/AzOps',
'Azure/terraform-azurerm-alz',
'Azure/aks-engine',
'Azure/vscode-aks-tools',
'Azure/iotedge-eflow',
'Azure/Bio-Compliancy',
'Azure/azure-cosmos-db-emulator-docker',
'Azure/go-autorest',
'Azure/NetworkMonitoring',
'Azure/migration',
'Azure/react-azure-maps',
'Azure/az-ps-module-versions',
'Azure/fetch-event-source',
'Azure/meta-iot-hub-device-update-delta',
'Azure/gitops-connector',
'Azure/sqlmi',
'Azure/custom-script-extension-linux',
'Azure/Test-Drive-Azure-Synapse-with-a-1-click-POC',
'Azure/Azure-Media-Services-Explorer',
'Azure/Verified-Telemetry',
'Azure/azure-event-hubs-go',
'Azure/confidential-computing-cvm',
'Azure/azure-devtestlab',
'Azure/azure-amqp-common-go',
'Azure/aad-auth-proxy',
'Azure/k8s-bake',
'Azure/DevOps-Self-Hosted',
'Azure/azure-sphere-tools',
'Azure/gitops-flux2-kustomize-helm-mt',
'Azure/WindowsVMAgent',
'Azure/spring-cloud-azure-tools',
'Azure/azure-diskinspect-service',
'Azure/dps-keygen',
'Azure/azure-docs-json-samples',
'Azure/azure-storage-common-php',
'Azure/ALAR',
'Azure/AnalyticsinaBox',
'Azure/homebrew-functions',
'Azure/iotedgedev',
'Azure/cyclecloud-gridengine',
'Azure/cyclecloud-pbspro',
'Azure/counterfit',
'Azure/azure-iot-dps-node',
'Azure/azure-batch-cli-extensions',
'Azure/fleet-networking',
'Azure/mlops-v2',
'Azure/SET',
'Azure/bicep-extensibility',
'Azure/KAN',
'Azure/sonic-build-tools',
'Azure/homebrew-aks-engine',
'Azure/ArcEnabledServersGroupPolicy',
'Azure/azure-odata-sql-js',
'Azure/Virtual-Machine-Restore-Points',
'Azure/AKS-Edge-Labs',
'Azure/HDInsight',
'Azure/Azure-Lighthouse-samples',
'Azure/azure-operation-script',
'Azure/container-upstream',
'Azure/amqpnetlite',
'Azure/azure-functions-language-worker-protobuf',
'Azure/azure-documentdb-changefeedprocessor-dotnet',
'Azure/Azure-Monitor-for-SAP-solutions-preview',
'Azure/go-ntlmssp',
'Azure/fta-live-iac',
'Azure/LabServices',
'Azure/azure-sdk-for-sap-odata',
'Azure/meta-raspberrypi-adu',
'Azure/meta-azure-device-update',
'Azure/WellArchitected-Tools',
'Azure/KeyVault-Secrets-Rotation-StorageAccount-PowerShell',
'Azure/AzureML-NLP',
'Azure/wordpress-linux-appservice',
'Azure/azure-migrate-discovery-extension-events',
'Azure/azure-migrate-export',
'Azure/powershell',
'Azure/powerautomate-avd-starter-kit',
'Azure/sql-action',
'Azure/kdebug',
'Azure/knarly',
'Azure/DotNetty',
'Azure/azure-relay-node',
'Azure/azure-functions-microsoftgraph-extension',
'Azure/mlops-workshop-code-public',
'Azure/docker-login',
'Azure/avshub',
'Azure/azure-data-lake-store-python',
'Azure/ato-toolkit',
'Azure/mlops-v2-workshop',
'Azure/aks-hybrid',
'Azure/azureml-managed-network-isolation',
'Azure/kafka-sink-azure-kusto',
'Azure/iotc-device-bridge',
'Azure/databox-adls-loader',
'Azure/msrest-for-python',
'Azure/Stormspotter',
'Azure/aci-deploy',
'Azure/mlops-v2-gha-demo',
'Azure/azure-kusto-rust',
'Azure/AML-Kubernetes',
'Azure/AppConfiguration-Sync',
'Azure/pipelines',
'Azure/azure-cosmos-cassandra-extensions',
'Azure/azure-kusto-trender',
'Azure/hpcpack-web-portal',
'Azure/awps-swa-sample',
'Azure/API-Management',
'Azure/-Microsoft-Defender-for-IoT',
'Azure/azure-sdk-for-node',
'Azure/AzMigrate-Hydration',
'Azure/homebrew-draft',
'Azure/sustainability',
'Azure/azure-synapse-analytics-end2end',
'Azure/caf-terraform-landingzones-platform-starter',
'Azure/apim-lab',
'Azure/azure-storage-ruby',
'Azure/ARO-Landing-Zone-Accelerator',
'Azure/benchpress',
'Azure/azure-managedapp-samples',
'Azure/azure-functions-integration-tests',
'Azure/aks-baseline-automation',
'Azure/azure-spatial-anchors-samples',
'Azure/azure-mysql',
'Azure/AI-PredictiveMaintenance',
'Azure/azure-event-hubs-c',
'Azure/fta-azure-machine-learning',
'Azure/rest-api-specs-scripts',
'Azure/hpcpack-acm-portal',
'Azure/logger-js',
'Azure/fabric-react-jsonschema-form',
'Azure/azure-storage-js',
'Azure/gocover',
'Azure/functions-container-action',
'Azure/policy-compliance-scan',
'Azure/mysql',
'Azure/postgresql',
'Azure/appservice-settings',
'Azure/azure-documentdb-datamigrationtool',
'Azure/osdu-bicep',
'Azure/elastic-db-tools',
'Azure/react-azure-maps-playground',
'Azure/azure-schema-registry-for-kafka',
'Azure/spring-boot-container-quickstart',
'Azure/AzOps-Accelerator',
'Azure/k8s-lint',
'Azure/redis-on-azure-workshop',
'Azure/API-Portal',
'Azure/opendigitaltwins-dtdl',
'Azure/azure-vmware-solution',
'Azure/kusto-adx-cse',
'Azure/iot-edge-testing-utility',
'Azure/keyvaultcertdownloader',
'Azure/Unreal-Pixel-Streaming',
'Azure/reddog-code',
'Azure/Synapse-workspace-deployment',
'Azure/dfs-namespace-cluster-examples',
'Azure/Batch',
'Azure/AzureDatabricksBestPractices',
'Azure/aks-set-context',
'Azure/azure-iot-connection-diagnostic-tool',
'Azure/AzureStor',
'Azure/Azure-Purview-API-PowerShell',
'Azure/azhpc-extensions',
'Azure/azure-functions-iothub-extension',
'Azure/azure-service-bus-java',
'Azure/azure-functions-eventgrid-extension',
'Azure/azure-websites-security',
'Azure/capacityreservationsharing',
'Azure/carbon-aware-keda-operator',
'Azure/kubeflow-aks',
'Azure/go-shuttle',
'Azure/ms-rest-js',
'Azure/missionlz-edge',
'Azure/azure-storage-mirror',
'Azure/AzureStack-Tools',
'Azure/azure-event-hubs',
'Azure/azure-iot-protocol-gateway',
'Azure/cyclecloud-marketplace-image',
'Azure/aca-java-runtimes-workshop',
'Azure/azure-internet-analyzer-java',
'Azure/azure-batch-rendering',
'Azure/iot-central-file-upload-device',
'Azure/mlops-templates',
'Azure/vld',
'Azure/medical-imaging',
'Azure/actions-workflow-samples',
'Azure/azure-sdk-korean',
'Azure/aks-advanced-autoscaling',
'Azure/orkestra',
'Azure/Vision-AI-DevKit-Pages',
'Azure/iotc-migrator',
'Azure/communication-services-pstn-calling',
'Azure/communication-monitoring',
'Azure/tflint-ruleset-azurerm-ext',
'Azure/aca-java-runtimes-workshop-template',
'Azure/codegenapp',
'Azure/git-rest-api',
'Azure/azure-amqp',
'Azure/Guest-Configuration-Extension',
'Azure/apim-landing-zone-accelerator',
'Azure/azure-storage-node',
'Azure/ADX-in-a-Day',
'Azure/NoOpsAccelerator',
'Azure/Cloud-Native',
'Azure/app-configuration-import-action',
'Azure/batch-insights',
'Azure/openapi-diff',
'Azure/cyclecloud-scalelib',
'Azure/ArcResourceBridge',
'Azure/dapr-workflows-aks-sample',
'Azure/azure-extension-foundation',
'Azure/fta-aas',
'Azure/arc-k8s-demo',
'Azure/azure-docs',
'Azure/Industrial-IoT-Gateway-Installer',
'Azure/azure-iot-connected-factory',
'Azure/sterling',
'Azure/deployment-what-if-action',
'Azure/Asteroid',
'Azure/azure-storage-php',
'Azure/config-driven-data-pipeline',
'Azure/azure-kusto-avro-conv',
'Azure/ai-solution-accelerators-list',
'Azure/data-factory-deploy-action',
'Azure/pytest-azurepipelines',
'Azure/azure-notificationhubs-ios',
'Azure/vscode-azureterraform',
'Azure/hpcpack-linux-agent',
'Azure/github-nginx-cache',
'Azure/container-apps-deploy-pipelines-task',
'Azure/fta-synapse-serverless-dacpac-builder',
'Azure/petabyte-scale-ai-data-lake-dashboard',
'Azure/azure-storage-file-php',
'Azure/azure-query-js',
'Azure/azure-resource-provider-sdk',
'Azure/envconf',
'Azure/azure-storage-net',
'Azure/azure-storage-java',
'Azure/azure-storage-cpp',
'Azure/oav-express',
'Azure/azure-websites-java-remote-debugging',
'Azure/node-inspector',
'Azure/azure-storage-android',
'Azure/azure-linux-automation',
'Azure/azure-documentdb-java',
'Azure/AzureQuickStartsProjects',
'Azure/azure-media-services-samples',
'Azure/azure-batch-apps-python',
'Azure/rtable',
'Azure/hdinsight-script-actions',
'Azure/azure-cli-docker',
'Azure/azure-mobile-engagement-cordova',
'Azure/azure-mobile-engagement-samples',
'Azure/azure-mobile-engagement-capptain-cordova',
'Azure/Azure-Apache-Migration-Tool',
'Azure/Azure-MachineLearning-ClientLibrary-R',
'Azure/identity-management-samples',
'Azure/azure-storage-ios',
'Azure/Azure-vpn-config-samples',
'Azure/azure-webjobs-quickstart',
'Azure/azure-insights-schemas',
'Azure/azure-support-scripts',
'Azure/azure-vm-scripts',
'Azure/azure-extensions-cli',
'Azure/parse-server-example',
'Azure/finance-analytics',
'Azure/hdinsight-phoenix-sharp']
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)
    
    
######### For initial analysis ###############
def basic_clean(text):
    '''
    take in a string and apply some basic text cleaning to it:
    * Lowercase everything
    * Normalize unicode characters
    * Replace anything that is not a letter, number, whitespace or a single quote.
    '''
    text = text.lower()  # Lowercase everything
    tedfxt = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # Normalize unicode characters
    text = re.sub(r"[^a-z0-9\s']", ' ', text)  # Replace anything that is not a letter, number, whitespace, or single quote
    return text

def tokenize(text):
    '''
    take in a string and tokenize all the words in the string
    '''
    tokenizer = ToktokTokenizer()
    return tokenizer.tokenize(text)


def stemmer(text):
    '''
    accept some text and return the text after applying stemming to all the words
    '''
    stemmer = nltk.stem.PorterStemmer()
    return ' '.join([stemmer.stemmer(word) for word in text.split()])


def lemmatize(text):
    '''
    accept some text and return the text after applying lemmatization to each word
    '''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_stopwords(text, extra_words=[], exclude_words=[]):
    '''
    accept some text and return the text after removing all the stopwords.
    This function defines two optional parameters, extra_words and exclude_words. These parameters define any additional stop words to include,
    and any words that we don't want to remove.
    '''
    ADDITIONAL_STOPWORDS = ['azure','http','com','github','microsoft']

    stopword_list = stopwords.words('english')+ADDITIONAL_STOPWORDS
    for word in extra_words:
        stopword_list.append(word)
    for word in exclude_words:
        stopword_list.remove(word)
    return ' '.join([word for word in text.split() if word not in stopword_list])



def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    ADDITIONAL_STOPWORDS = ['azure','http','com','github','microsoft']
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

def join_words(text):
    '''
    used by wrangle_git to join cleaned column
    '''
    return ' '.join(text)

def wrangle_git():
    '''
    this function wrangles the github.csv for use in initial exploration
    it applies general data cleanup principles 
    '''
    git_df = pd.read_csv('github.csv',index_col=0)
    git_df = pd.DataFrame(git_df)

    # remove unwanted languages
    l = ['C#', 'PowerShell', 'Go', 'TypeScript', 'Python', 'JavaScript']
    git_df.language = np.where(git_df.language.isin(l), git_df.language, np.nan)
    # drop nulls
    git_df = git_df.dropna()
    #clean and lemmatize
    git_df['clean'] = git_df['readme_contents'].apply(basic_clean).apply(tokenize).apply(lambda x: ' '.join(x))
    lemmatizer = WordNetLemmatizer()
    git_df['lemmatized'] = git_df['clean'].apply(tokenize).apply(lambda x: [lemmatizer.lemmatize(word) for word in x]).apply(lambda x: ' '.join(x))
    # applies clean function and adds column
    git_df['clean_lem']= (git_df.lemmatized.apply(clean))
    git_df.clean_lem = git_df.clean_lem.apply(lambda x : join_words(x))
    git_df['word_count'] = git_df['clean_lem'].apply(lambda x: len(x.split()))
    return git_df

