import requests
import io

class GraphdbUploader:
  @staticmethod
  def upload(file_name):
    url = f"http://localhost:7200/repositories/AMD"
    headers = {
      "Content-Type": 'text/turtle',
    }
    data = GraphdbUploader.config_file("AMD", "Repository Description")
    requests.delete(url)
    response = requests.put(url, headers=headers, data=data)
    return response

  @staticmethod
  def config_file(repo_id, repo_description):
    config = f"""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep: <http://www.openrdf.org/config/repository#> .
@prefix sail: <http://www.openrdf.org/config/sail#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<rdf:{repo_id}> a rep:Repository;
    rep:repositoryID "{repo_id}";
    rep:repositoryImpl [
        rep:repositoryType "graphdb:SailRepository";
        <http://www.openrdf.org/config/repository/sail#sailImpl> [
            <http://www.ontotext.com/config/graphdb#base-URL> "http://localhost:7200/{repo_id}#";
            <http://www.ontotext.com/config/graphdb#check-for-inconsistencies> "false";
            <http://www.ontotext.com/config/graphdb#defaultNS> "http://localhost:7200/{repo_id}#";
            <http://www.ontotext.com/config/graphdb#disable-sameAs> "false";
            <http://www.ontotext.com/config/graphdb#enable-context-index> "false";
            <http://www.ontotext.com/config/graphdb#enable-fts-index> "false";
            <http://www.ontotext.com/config/graphdb#enable-literal-index> "true";
            <http://www.ontotext.com/config/graphdb#enablePredicateList> "true";
            <http://www.ontotext.com/config/graphdb#entity-id-size> "32";
            <http://www.ontotext.com/config/graphdb#entity-index-size> "10000000";
            <http://www.ontotext.com/config/graphdb#fts-indexes> ("default" "iri");
            <http://www.ontotext.com/config/graphdb#fts-iris-index> "none";
            <http://www.ontotext.com/config/graphdb#fts-string-literals-index> "default";
            <http://www.ontotext.com/config/graphdb#imports> "/app/ontologies/{repo_id}.rdf";
            <http://www.ontotext.com/config/graphdb#in-memory-literal-properties> "true";
            <http://www.ontotext.com/config/graphdb#query-limit-results> "0";
            <http://www.ontotext.com/config/graphdb#query-timeout> "0";
            <http://www.ontotext.com/config/graphdb#read-only> "false";
            <http://www.ontotext.com/config/graphdb#repository-type> "file-repository";
            <http://www.ontotext.com/config/graphdb#ruleset> "rdfsplus-optimized";
            <http://www.ontotext.com/config/graphdb#storage-folder> "storage";
            <http://www.ontotext.com/config/graphdb#throw-QueryEvaluationException-on-timeout>
            "false";
            sail:sailType "graphdb:Sail"
        ]
    ];
    rdfs:label "{repo_description}" ."""

    return config
