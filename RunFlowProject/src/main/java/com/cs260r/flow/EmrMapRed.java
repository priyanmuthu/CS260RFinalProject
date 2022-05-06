package com.cs260r.flow;

import java.io.IOException;

import com.amazonaws.AmazonClientException;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduce;
import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduceClientBuilder;
import com.amazonaws.services.elasticmapreduce.model.StepConfig;
import com.amazonaws.services.elasticmapreduce.util.StepFactory;

public class EmrMapRed {
    public static void main(String[] args) {

        // Initial code from: https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-common-programming-sample.html
        // Action on failure: TERMINATE_JOB_FLOW, TERMINATE_CLUSTER, CANCEL_AND_WAIT, and CONTINUE

        System.out.println("Hello!");
        AWSCredentials credentials_profile = null;
        AWSCredentialsProvider credentials_provider = null;
        try{
            credentials_provider = new ProfileCredentialsProvider("default");
            credentials_profile = credentials_provider.getCredentials();
        } catch (Exception e){
            throw new AmazonClientException(
                    "Cannot load credentials from .aws/credentials file. " +
                            "Make sure that the credentials file exists and the profile name is specified within it.",
                    e);
        }

        // AWS EMR cluster
        AmazonElasticMapReduce emr = AmazonElasticMapReduceClientBuilder.standard()
                .withCredentials(credentials_provider)
                .withRegion(Regions.US_WEST_1)
                .build();

        // Run a bash script using a predefined step in the StepFactory helper class
        StepFactory stepFactory = new StepFactory();
//        StepConfig runBashScriptStep = new StepConfig()
//                .withName("Run Bash Script")
//                .withHadoopJarStep(stepFactory.newScriptRunnerStep("s3://jeffgoll/emr-scripts/create_users.sh"))
//                .withActionOnFailure("TERMINATE_JOB_FLOW")
//                .withHadoopJarStep(stepFactory.newEnableDebuggingStep());



        // Steps:
        // Initialize cluster, if not exists
        // Run bash script to copy the JAR files and input files to the cluster
        // Modify the yarn-site.xml to use the custom scheduler
        // Run the MapReduce job


    }
}
