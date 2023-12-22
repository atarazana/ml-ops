{{/*
Expand the name of the chart.
*/}}
{{- define "ml.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ml.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ml.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml.labels" -}}
helm.sh/chart: {{ include "ml.chart" . }}
{{/*{{ include "ml.selectorLabels" . }}*/}}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-ui.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml.name" . }}-ui
app.kubernetes.io/instance: {{ .Release.Name }}-ui
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-bk.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml.name" . }}-bk
app.kubernetes.io/instance: {{ .Release.Name }}-bk
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ml.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
