package com.cs260r.flow;

import java.io.IOException;

import com.amazonaws.AmazonClientException;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;

public class EmrMapRed {
    public static void main(String[] args) {
        System.out.println("Hello!");
        AWSCredentials credentials_profile = null;
        try{
            credentials_profile = new ProfileCredentialsProvider("default").getCredentials();
        } catch (Exception e){
            throw new AmazonClientException(
                    "Cannot load credentials from .aws/credentials file. " +
                            "Make sure that the credentials file exists and the profile name is specified within it.",
                    e);
        }

        // EMR class
    }
}
