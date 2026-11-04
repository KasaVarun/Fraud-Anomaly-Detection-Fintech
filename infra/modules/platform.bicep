@description('Deployment environment.')
@minLength(2)
param environmentName string
param location string
@minLength(4)
param uniqueSuffix string
param tags object
param allowedOrigins array
param postgresAdministratorLogin string
@secure()
param postgresAdministratorPassword string
param entraTenantId string
param entraApplicationId string
param containerImageDigest string
param deployContainerApp bool
param workerImageDigest string
param deployPipelineJobs bool
param artifactManifestBlob string
param artifactManifestSha256 string
param stagingRetentionDays int
param containerRegistrySku string
param containerMinReplicas int
param containerMaxReplicas int
param logRetentionDays int
param postgresSkuName string
param postgresStorageGiB int
param postgresBackupRetentionDays int
param postgresExtensions array

var baseName = 'fraud-${environmentName}-${uniqueSuffix}'
var compactName = toLower(replace(baseName, '-', ''))
var postgresName = take('${baseName}-pg', 63)
var noSqlName = take('${compactName}nosql', 44)
var gremlinName = take('${compactName}gremlin', 44)
var vaultName = take('${baseName}-kv', 24)
var registryName = take('fraud${uniqueSuffix}acr', 50)
var stagingName = take('${compactName}stage', 24)
var identityName = '${baseName}-api-id'
var appName = '${baseName}-api'
var jobPrefix = take('fd-${environmentName}-${uniqueSuffix}', 15)
var databaseName = 'fraud_detection'
var rawContainerName = 'transactions_raw'
var embeddingContainerName = 'behavior_embeddings'
var gremlinDatabaseName = 'fraud_graph'
var graphName = 'evidence_graph'

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-01-01' = {
  name: '${baseName}-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: ['10.42.0.0/16']
    }
    subnets: [
      {
        name: 'container-apps'
        properties: {
          addressPrefix: '10.42.0.0/23'
          delegations: [
            {
              name: 'Microsoft.App.environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: 'postgresql'
        properties: {
          addressPrefix: '10.42.2.0/24'
          delegations: [
            {
              name: 'Microsoft.DBforPostgreSQL.flexibleServers'
              properties: {
                serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
              }
            }
          ]
        }
      }
      {
        name: 'private-endpoints'
        properties: {
          addressPrefix: '10.42.3.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'power-bi-gateway'
        properties: {
          addressPrefix: '10.42.4.0/27'
          delegations: [
            {
              name: 'Microsoft.PowerPlatform.vnetaccesslinks'
              properties: {
                serviceName: 'Microsoft.PowerPlatform/vnetaccesslinks'
              }
            }
          ]
        }
      }
    ]
  }
}

resource containerSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-01-01' existing = {
  name: 'container-apps'
  parent: virtualNetwork
}

resource postgresSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-01-01' existing = {
  name: 'postgresql'
  parent: virtualNetwork
}

resource endpointSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-01-01' existing = {
  name: 'private-endpoints'
  parent: virtualNetwork
}

resource postgresDns 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'private.postgres.database.azure.com'
  location: 'global'
  tags: tags
}

resource documentsDns 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.documents.azure.com'
  location: 'global'
  tags: tags
}

resource gremlinDns 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.gremlin.cosmos.azure.com'
  location: 'global'
  tags: tags
}

resource vaultDns 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.vaultcore.azure.net'
  location: 'global'
  tags: tags
}

resource blobDns 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.blob.${az.environment().suffixes.storage}'
  location: 'global'
  tags: tags
}

resource postgresDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  name: '${baseName}-postgres-link'
  parent: postgresDns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource documentsDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  name: '${baseName}-documents-link'
  parent: documentsDns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: virtualNetwork.id }
  }
}

resource gremlinDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  name: '${baseName}-gremlin-link'
  parent: gremlinDns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: virtualNetwork.id }
  }
}

resource vaultDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  name: '${baseName}-vault-link'
  parent: vaultDns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: virtualNetwork.id }
  }
}

resource blobDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  name: '${baseName}-blob-link'
  parent: blobDns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: virtualNetwork.id }
  }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${baseName}-logs'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: logRetentionDays
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${baseName}-insights'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
    IngestionMode: 'LogAnalytics'
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: registryName
  location: location
  tags: tags
  sku: {
    name: containerRegistrySku
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    policies: {
      retentionPolicy: {
        days: 7
        status: 'enabled'
      }
      trustPolicy: {
        type: 'Notary'
        status: 'disabled'
      }
    }
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

resource registryPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, 'AcrPull')
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  tags: tags
  properties: {
    tenantId: entraTenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enablePurgeProtection: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Disabled'
  }
}

resource vaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identity.id, 'KeyVaultSecretsUser')
  scope: vault
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
  }
}

resource vaultEndpoint 'Microsoft.Network/privateEndpoints@2024-01-01' = {
  name: '${baseName}-kv-pe'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: endpointSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'vault'
        properties: {
          privateLinkServiceId: vault.id
          groupIds: ['vault']
        }
      }
    ]
  }
}

resource vaultZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-01-01' = {
  name: 'default'
  parent: vaultEndpoint
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'vault'
        properties: {
          privateDnsZoneId: vaultDns.id
        }
      }
    ]
  }
}

resource staging 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: stagingName
  location: location
  tags: union(tags, { purpose: 'operational-artifact-staging' })
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    minimumTlsVersion: 'TLS1_2'
    publicNetworkAccess: 'Disabled'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  name: 'default'
  parent: staging
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource artifactContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'deployment-artifacts'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource stagingLifecycle 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  name: 'default'
  parent: staging
  properties: {
    policy: {
      rules: [
        {
          name: 'expire-operational-staging'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: { blobTypes: ['blockBlob'] }
            actions: {
              baseBlob: { delete: { daysAfterModificationGreaterThan: stagingRetentionDays } }
              snapshot: { delete: { daysAfterCreationGreaterThan: 7 } }
              version: { delete: { daysAfterCreationGreaterThan: 7 } }
            }
          }
        }
      ]
    }
  }
}

resource stagingReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(staging.id, identity.id, 'StorageBlobDataContributor')
  scope: staging
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}

resource stagingEndpoint 'Microsoft.Network/privateEndpoints@2024-01-01' = {
  name: '${baseName}-staging-pe'
  location: location
  tags: tags
  properties: {
    subnet: { id: endpointSubnet.id }
    privateLinkServiceConnections: [
      {
        name: 'blob'
        properties: {
          privateLinkServiceId: staging.id
          groupIds: ['blob']
        }
      }
    ]
  }
}

resource stagingZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-01-01' = {
  name: 'default'
  parent: stagingEndpoint
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob'
        properties: { privateDnsZoneId: blobDns.id }
      }
    ]
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: postgresName
  location: location
  tags: tags
  sku: {
    name: postgresSkuName
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: postgresAdministratorLogin
    administratorLoginPassword: postgresAdministratorPassword
    version: '16'
    storage: {
      storageSizeGB: postgresStorageGiB
      autoGrow: 'Disabled'
    }
    backup: {
      backupRetentionDays: postgresBackupRetentionDays
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Enabled'
      tenantId: entraTenantId
    }
    network: {
      delegatedSubnetResourceId: postgresSubnet.id
      privateDnsZoneArmResourceId: postgresDns.id
      publicNetworkAccess: 'Disabled'
    }
  }
  dependsOn: [postgresDnsLink]
}

resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  name: databaseName
  parent: postgres
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource postgresExtensionsConfig 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = if (length(postgresExtensions) > 0) {
  name: 'azure.extensions'
  parent: postgres
  properties: {
    source: 'user-override'
    value: join(postgresExtensions, ',')
  }
}

resource postgresPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: 'postgres-admin-password'
  parent: vault
  properties: {
    value: postgresAdministratorPassword
    contentType: 'password'
  }
  dependsOn: [vaultZoneGroup]
}

resource noSql 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = {
  name: noSqlName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      { name: 'EnableServerless' }
      { name: 'EnableNoSQLVectorSearch' }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    publicNetworkAccess: 'Disabled'
    disableLocalAuth: true
    minimalTlsVersion: 'Tls12'
  }
}

resource sqlDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-11-15' = {
  name: 'fraud_evidence'
  parent: noSql
  properties: {
    resource: {
      id: 'fraud_evidence'
    }
  }
}

resource noSqlDataContributor 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-11-15' = {
  name: guid(noSql.id, identity.id, 'CosmosDataContributor')
  parent: noSql
  properties: {
    principalId: identity.properties.principalId
    roleDefinitionId: '${noSql.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    scope: noSql.id
  }
}

resource rawContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  name: rawContainerName
  parent: sqlDatabase
  properties: {
    resource: {
      id: rawContainerName
      partitionKey: {
        paths: ['/sender_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/forensic_balances/*' }]
      }
    }
  }
}

resource embeddingContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-11-15' = {
  name: embeddingContainerName
  parent: sqlDatabase
  properties: {
    resource: {
      id: embeddingContainerName
      partitionKey: {
        paths: ['/sender_id']
        kind: 'Hash'
        version: 2
      }
      vectorEmbeddingPolicy: {
        vectorEmbeddings: [
          {
            path: '/vector'
            dataType: 'float32'
            distanceFunction: 'cosine'
            dimensions: 384
          }
        ]
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/vector/*' }]
        vectorIndexes: [
          {
            path: '/vector'
            type: 'quantizedFlat'
          }
        ]
      }
    }
  }
}

resource noSqlEndpoint 'Microsoft.Network/privateEndpoints@2024-01-01' = {
  name: '${baseName}-nosql-pe'
  location: location
  tags: tags
  properties: {
    subnet: { id: endpointSubnet.id }
    privateLinkServiceConnections: [
      {
        name: 'sql'
        properties: {
          privateLinkServiceId: noSql.id
          groupIds: ['Sql']
        }
      }
    ]
  }
}

resource noSqlZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-01-01' = {
  name: 'default'
  parent: noSqlEndpoint
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'documents'
        properties: { privateDnsZoneId: documentsDns.id }
      }
    ]
  }
}

resource gremlin 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = {
  name: gremlinName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      { name: 'EnableGremlin' }
      { name: 'EnableServerless' }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    publicNetworkAccess: 'Disabled'
    disableLocalAuth: false
    minimalTlsVersion: 'Tls12'
  }
}

resource gremlinDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2024-11-15' = {
  name: gremlinDatabaseName
  parent: gremlin
  properties: {
    resource: { id: gremlinDatabaseName }
  }
}

resource graph 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases/graphs@2024-11-15' = {
  name: graphName
  parent: gremlinDatabase
  properties: {
    resource: {
      id: graphName
      partitionKey: {
        paths: ['/pk']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [{ path: '/*' }]
        excludedPaths: []
      }
    }
  }
}

resource gremlinEndpoint 'Microsoft.Network/privateEndpoints@2024-01-01' = {
  name: '${baseName}-gremlin-pe'
  location: location
  tags: tags
  properties: {
    subnet: { id: endpointSubnet.id }
    privateLinkServiceConnections: [
      {
        name: 'gremlin'
        properties: {
          privateLinkServiceId: gremlin.id
          groupIds: ['Gremlin']
        }
      }
    ]
  }
}

resource gremlinZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-01-01' = {
  name: 'default'
  parent: gremlinEndpoint
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'gremlin'
        properties: { privateDnsZoneId: gremlinDns.id }
      }
    ]
  }
}

resource gremlinKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: 'cosmos-gremlin-key'
  parent: vault
  properties: {
    value: gremlin.listKeys().primaryMasterKey
    contentType: 'Cosmos DB Gremlin key'
  }
  dependsOn: [gremlinZoneGroup, vaultZoneGroup]
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${baseName}-cae'
  location: location
  tags: tags
  properties: {
    vnetConfiguration: {
      infrastructureSubnetId: containerSubnet.id
      internal: false
    }
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var postgresDsn = 'postgresql+psycopg://${postgresAdministratorLogin}:${postgresAdministratorPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/${databaseName}?sslmode=require'

resource postgresDsnSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: 'postgres-dsn'
  parent: vault
  properties: {
    value: postgresDsn
    contentType: 'PostgreSQL TLS DSN'
  }
  dependsOn: [postgresPasswordSecret]
}

var commonEnvironment = [
  { name: 'FRAUD_ENVIRONMENT', value: 'production' }
  { name: 'FRAUD_HOST', value: '0.0.0.0' }
  { name: 'FRAUD_PORT', value: '8000' }
  { name: 'FRAUD_LOG_LEVEL', value: 'INFO' }
  { name: 'FRAUD_AUTH_MODE', value: 'entra' }
  { name: 'FRAUD_ENTRA_TENANT_ID', value: entraTenantId }
  { name: 'FRAUD_ENTRA_AUDIENCE', value: 'api://${entraApplicationId}' }
  { name: 'FRAUD_ENTRA_ISSUER', value: '${az.environment().authentication.loginEndpoint}${entraTenantId}/v2.0' }
  { name: 'FRAUD_ENTRA_JWKS_URL', value: '${az.environment().authentication.loginEndpoint}${entraTenantId}/discovery/v2.0/keys' }
  { name: 'FRAUD_ENTRA_REQUIRED_SCOPE', value: 'Fraud.Investigate' }
  { name: 'FRAUD_CORS_ALLOWED_ORIGINS', value: string(allowedOrigins) }
  { name: 'FRAUD_COSMOS_NOSQL_ENDPOINT', value: noSql.properties.documentEndpoint }
  { name: 'FRAUD_COSMOS_USE_MANAGED_IDENTITY', value: 'true' }
  { name: 'FRAUD_AZURE_MANAGED_IDENTITY_CLIENT_ID', value: identity.properties.clientId }
  { name: 'FRAUD_COSMOS_NOSQL_DATABASE', value: sqlDatabase.name }
  { name: 'FRAUD_COSMOS_RAW_CONTAINER', value: rawContainer.name }
  { name: 'FRAUD_COSMOS_EMBEDDING_CONTAINER', value: embeddingContainer.name }
  { name: 'FRAUD_GREMLIN_ENDPOINT', value: 'wss://${gremlin.name}.gremlin.cosmos.azure.com:443/' }
  { name: 'FRAUD_GREMLIN_USERNAME', value: '/dbs/${gremlinDatabase.name}/colls/${graph.name}' }
  { name: 'FRAUD_GREMLIN_DATABASE', value: gremlinDatabase.name }
  { name: 'FRAUD_GREMLIN_GRAPH', value: graph.name }
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
  { name: 'FRAUD_POSTGRES_DSN', secretRef: 'postgres-dsn' }
  { name: 'FRAUD_GREMLIN_KEY', secretRef: 'cosmos-gremlin-key' }
]

var workerEnvironment = concat(commonEnvironment, [
  { name: 'FRAUD_WORK_ROOT', value: '/work' }
  { name: 'FRAUD_STAGING_ACCOUNT_URL', value: staging.properties.primaryEndpoints.blob }
  { name: 'FRAUD_STAGING_CONTAINER', value: artifactContainer.name }
  { name: 'FRAUD_ARTIFACT_MANIFEST_BLOB', value: artifactManifestBlob }
  { name: 'FRAUD_ARTIFACT_MANIFEST_SHA256', value: artifactManifestSha256 }
  { name: 'FRAUD_BEHAVIORAL_TEMPLATE_ARTIFACT', value: '/work/data/processed/behavior_templates.parquet' }
  { name: 'FRAUD_EMBEDDING_LOCAL_ARTIFACT', value: '/work/data/processed/behavior_embeddings.parquet' }
  { name: 'FRAUD_MODEL_FEATURE_ARTIFACT', value: '/work/data/processed/model_features_v2.parquet' }
  { name: 'FRAUD_MODEL_ARTIFACT_ROOT', value: '/work/artifacts/models/releases' }
  { name: 'FRAUD_EMBEDDING_MODEL_CACHE', value: '/work/artifacts/models/sentence-transformers' }
  { name: 'FRAUD_INCREMENTAL_ARTIFACT_ROOT', value: '/work/artifacts/incremental/phase8' }
  { name: 'FRAUD_INCREMENTAL_DATA_ROOT', value: '/work/data/processed/phase8' }
])

var pipelineJobDefinitions = [
  { suffix: 'migrate', stage: 'schema-migration', timeout: 900, retries: 1, cpu: '0.5', memory: '1Gi' }
  { suffix: 'initial-load', stage: 'initial-relational-load', timeout: 3600, retries: 1, cpu: '1.0', memory: '2Gi' }
  { suffix: 'raw-project', stage: 'raw-projection', timeout: 7200, retries: 1, cpu: '1.0', memory: '2Gi' }
  { suffix: 'graph-project', stage: 'graph-projection', timeout: 7200, retries: 1, cpu: '1.0', memory: '2Gi' }
  { suffix: 'embeddings', stage: 'behavioral-embeddings', timeout: 14400, retries: 0, cpu: '2.0', memory: '4Gi' }
  { suffix: 'graph-snapshot', stage: 'graph-snapshot', timeout: 14400, retries: 0, cpu: '2.0', memory: '4Gi' }
  { suffix: 'feature-model', stage: 'feature-model', timeout: 14400, retries: 0, cpu: '2.0', memory: '4Gi' }
  { suffix: 'incremental', stage: 'incremental', timeout: 7200, retries: 0, cpu: '2.0', memory: '4Gi' }
  { suffix: 'reconcile', stage: 'reconciliation', timeout: 3600, retries: 0, cpu: '1.0', memory: '2Gi' }
  { suffix: 'evaluate', stage: 'evaluation', timeout: 3600, retries: 0, cpu: '1.0', memory: '2Gi' }
  { suffix: 'publish', stage: 'publication', timeout: 1800, retries: 0, cpu: '0.5', memory: '1Gi' }
]

resource app 'Microsoft.App/containerApps@2024-03-01' = if (deployContainerApp) {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        allowInsecure: false
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: [
        { name: 'postgres-dsn', keyVaultUrl: '${vault.properties.vaultUri}secrets/postgres-dsn', identity: identity.id }
        { name: 'cosmos-gremlin-key', keyVaultUrl: '${vault.properties.vaultUri}secrets/cosmos-gremlin-key', identity: identity.id }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImageDigest
          env: commonEnvironment
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          probes: [
            { type: 'Liveness', httpGet: { path: '/health/live', port: 8000 }, initialDelaySeconds: 10, periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 3 }
            { type: 'Readiness', httpGet: { path: '/health/ready', port: 8000 }, initialDelaySeconds: 10, periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 3 }
          ]
        }
      ]
      scale: {
        minReplicas: containerMinReplicas
        maxReplicas: containerMaxReplicas
      }
    }
  }
  dependsOn: [registryPull, vaultSecretsUser, noSqlDataContributor, postgresDsnSecret, gremlinKeySecret]
}

resource pipelineJobs 'Microsoft.App/jobs@2024-03-01' = [for job in pipelineJobDefinitions: if (deployPipelineJobs) {
  name: '${jobPrefix}-${job.suffix}'
  location: location
  tags: union(tags, { pipelineStage: job.stage })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    environmentId: environment.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: job.timeout
      replicaRetryLimit: job.retries
      registries: [{ server: registry.properties.loginServer, identity: identity.id }]
      secrets: [
        { name: 'postgres-dsn', keyVaultUrl: '${vault.properties.vaultUri}secrets/postgres-dsn', identity: identity.id }
        { name: 'cosmos-gremlin-key', keyVaultUrl: '${vault.properties.vaultUri}secrets/cosmos-gremlin-key', identity: identity.id }
      ]
      manualTriggerConfig: { parallelism: 1, replicaCompletionCount: 1 }
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: workerImageDigest
          args: [job.stage, '--timeout-seconds', string(job.timeout)]
          env: workerEnvironment
          resources: { cpu: json(job.cpu), memory: job.memory }
        }
      ]
    }
  }
  dependsOn: [registryPull, vaultSecretsUser, noSqlDataContributor, stagingReader, stagingZoneGroup, postgresDsnSecret, gremlinKeySecret]
}]

output registryName string = registry.name
output containerAppName string = deployContainerApp ? app.name : appName
output postgresServerName string = postgres.name
output noSqlAccountName string = noSql.name
output gremlinAccountName string = gremlin.name
output keyVaultName string = vault.name
output managedIdentityName string = identity.name
output stagingAccountName string = staging.name
output pipelineJobNames array = [for job in pipelineJobDefinitions: '${jobPrefix}-${job.suffix}']
