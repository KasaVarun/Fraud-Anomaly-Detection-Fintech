targetScope = 'subscription'

@description('Short environment name such as dev, test, or demo.')
@minLength(2)
@maxLength(12)
param environmentName string

@description('Azure region selected only after Gate A service/price validation.')
param location string

@description('Explicit resource group to create during approved Gate B provisioning.')
param resourceGroupName string

@description('Globally unique lowercase suffix used for globally scoped names.')
@minLength(4)
@maxLength(12)
param uniqueSuffix string

@description('Human owner tag used for cost and teardown accountability.')
param owner string

@description('ISO date after which demo resources should be reviewed for teardown.')
param expirationDate string

@description('Additional non-secret resource tags.')
param additionalTags object = {}

@description('Exact browser origins allowed by the read-only API.')
param allowedOrigins array = []

@description('PostgreSQL administrator login name; never use a personal identity.')
param postgresAdministratorLogin string

@secure()
@description('Gate B secret only. Never place a value in a committed parameter file.')
param postgresAdministratorPassword string

@description('Microsoft Entra tenant ID approved for the deployed API.')
param entraTenantId string

@description('Microsoft Entra application/client ID approved for the deployed API.')
param entraApplicationId string

@description('Immutable API image reference containing @sha256:, already pushed to the Gate B registry.')
param containerImageDigest string = ''

@description('Deploy the API only after its image exists in ACR.')
param deployContainerApp bool = false

@description('Immutable worker image reference containing @sha256:, already pushed to the Gate B registry.')
param workerImageDigest string = ''

@description('Create all manually triggered private pipeline jobs.')
param deployPipelineJobs bool = false

@description('Private staging blob containing the deployment-specific artifact manifest.')
param artifactManifestBlob string = 'deployment/artifact-manifest.json'

@description('SHA-256 of the deployment-specific artifact manifest.')
param artifactManifestSha256 string = ''

@description('Delete operational staging blobs after this many days.')
@minValue(7)
@maxValue(90)
param stagingRetentionDays int = 30

@allowed(['Basic', 'Standard'])
param containerRegistrySku string = 'Basic'

@description('Minimum Container Apps replicas. Zero minimizes demo idle cost.')
@minValue(0)
@maxValue(3)
param containerMinReplicas int = 0

@description('Maximum Container Apps replicas for the bounded investigator API.')
@minValue(1)
@maxValue(5)
param containerMaxReplicas int = 1

@description('Log Analytics retention in days.')
@minValue(30)
@maxValue(90)
param logRetentionDays int = 30

@description('Minimal PostgreSQL burstable SKU; review price before Gate B.')
param postgresSkuName string = 'Standard_B1ms'

@description('PostgreSQL storage size in GiB.')
@minValue(32)
@maxValue(128)
param postgresStorageGiB int = 32

@description('PostgreSQL backup retention in days.')
@minValue(7)
@maxValue(35)
param postgresBackupRetentionDays int = 7

@description('Existing approved PostgreSQL extensions. Empty because the current schema requires none.')
param postgresExtensions array = []

var tags = union(additionalTags, {
  project: 'fraud-anomaly-detection-fintech'
  environment: environmentName
  owner: owner
  expiration: expirationDate
  managedBy: 'bicep'
  gate: 'phase11-gate-b'
})

resource resourceGroup 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module resources 'modules/platform.bicep' = {
  name: 'fraud-platform-${environmentName}'
  scope: resourceGroup
  params: {
    environmentName: environmentName
    location: location
    uniqueSuffix: uniqueSuffix
    tags: tags
    allowedOrigins: allowedOrigins
    postgresAdministratorLogin: postgresAdministratorLogin
    postgresAdministratorPassword: postgresAdministratorPassword
    entraTenantId: entraTenantId
    entraApplicationId: entraApplicationId
    containerImageDigest: containerImageDigest
    deployContainerApp: deployContainerApp
    workerImageDigest: workerImageDigest
    deployPipelineJobs: deployPipelineJobs
    artifactManifestBlob: artifactManifestBlob
    artifactManifestSha256: artifactManifestSha256
    stagingRetentionDays: stagingRetentionDays
    containerRegistrySku: containerRegistrySku
    containerMinReplicas: containerMinReplicas
    containerMaxReplicas: containerMaxReplicas
    logRetentionDays: logRetentionDays
    postgresSkuName: postgresSkuName
    postgresStorageGiB: postgresStorageGiB
    postgresBackupRetentionDays: postgresBackupRetentionDays
    postgresExtensions: postgresExtensions
  }
}

output resourceGroupName string = resourceGroup.name
output registryName string = resources.outputs.registryName
output containerAppName string = resources.outputs.containerAppName
output postgresServerName string = resources.outputs.postgresServerName
output noSqlAccountName string = resources.outputs.noSqlAccountName
output gremlinAccountName string = resources.outputs.gremlinAccountName
output keyVaultName string = resources.outputs.keyVaultName
output managedIdentityName string = resources.outputs.managedIdentityName
output stagingAccountName string = resources.outputs.stagingAccountName
output pipelineJobNames array = resources.outputs.pipelineJobNames
