package com.mj.refund;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RefundApplication {

	public static void main(String[] args) {
		System.out.println("Welcome to the IRS Refund application");
        SpringApplication.run(RefundApplication.class, args);

	}

}
