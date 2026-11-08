---
title: "Step by step configuration of Hibernate, Spring MVC, Primefaces"
date: "2013-10-30T15:08:00+05:30"
recovered_from: "web.archive.org (snapshot 20201108133841)"
---

So, you have decided to use Spring, Hibernate and Primefaces in your
application. I am very much fond of each one of them cause, all of them very
easy to use and open source. I use all of them in my own application.<o:p>&lt;/o:p&gt;

But, the first time I tried to configure them in my application, I had
nightmare. The most tough part was to configure Hibernate with Spring and to find how to use Spring beans in Managed Beans. But now my application is stable and development has become easy
for me now.<o:p>&lt;/o:p&gt;

So, I thought of sharing the knowledge to configure the application to
use all of them.<o:p>&lt;/o:p&gt;

I will use HSQL as DB for this post. I assume that you are capable of
configuring your DB in the configuration files. Because, you are reading this
post means, you are an advanced Java Developer.<o:p>&lt;/o:p&gt;

**Prerequisite<o:p>&lt;/o:p&gt;

1.     You have knowledge on
Spring, Hibernate and JSF and know the basics of setting them up individually<o:p>&lt;/o:p&gt;

2.     Knowledge of Eclipse
IDE or Maven or any other build tool<o:p>&lt;/o:p&gt;

3.     Working database<o:p>&lt;/o:p&gt;

4.     Hibernate 4 and Spring
4 (For another pair of these, generally these configuration needs a little
change in the session factory configuration, which you can figure out in ease)<o:p>&lt;/o:p&gt;

5.     Primefaces (3 or 4)
and Spring(3 or 4)

This pair can have any combination. Choose any. Spring versioons are compatible
with Primefaces versions.<o:p>&lt;/o:p&gt;

**Configuration<o:p>&lt;/o:p&gt;

 I will go step by step for the configurations.<o:p>&lt;/o:p&gt;
**1. Mention the required
libraries in your build - Required
library list is huge cause Spring, Hibernate and Primefaces depend on
different libraries. So please be careful otherwise you will get many
exceptions. There are total 28 dependencies in total.

§  antlr-2.7.7.jar<o:p>&lt;/o:p&gt;

§  commons-logging-1.1.1.jar<o:p>&lt;/o:p&gt;

§  dom4j-1.6.1.jar<o:p>&lt;/o:p&gt;

§  el-impl-2.2.1-b05.jar<o:p>&lt;/o:p&gt;

§  hibernate-commons-annotations-4.0.5.Final.jar<o:p>&lt;/o:p&gt;

§  hibernate-core-4.3.7.Final.jar<o:p>&lt;/o:p&gt;

§  hibernate-jpa-2.1-api-1.0.0.Final.jar<o:p>&lt;/o:p&gt;

§  hsqldb.jar<o:p>&lt;/o:p&gt;

§  jandex-1.1.0.Final.jar<o:p>&lt;/o:p&gt;

§  javassist-3.18.1-GA.jar<o:p>&lt;/o:p&gt;

§  jboss-logging-3.1.3.GA.jar<o:p>&lt;/o:p&gt;

§  jboss-logging-annotations-1.2.0.Beta1.jar<o:p>&lt;/o:p&gt;

§  jboss-transaction-api_1.2_spec-1.0.0.Final.jar<o:p>&lt;/o:p&gt;

§  jsf-api-2.2.8-02.jar<o:p>&lt;/o:p&gt;

§  jsf-impl-2.2.8-02.jar<o:p>&lt;/o:p&gt;

§  jstl-1.2.jar<o:p>&lt;/o:p&gt;

§  primefaces-4.0.jar<o:p>&lt;/o:p&gt;

§  spring-aop-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-beans-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-context-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-context-support-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-core-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-expression-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-jdbc-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-orm-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-tx-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-web-4.1.3.RELEASE.jar<o:p>&lt;/o:p&gt;

§  spring-webmvc-4.1.3.RELEASE.jarWith all these dependencies, WebContent/WEB-INF/lib looks like<o:p>&lt;/o:p&gt;

![image](http://2.bp.blogspot.com/-TLdo1FZ1F1A/VIq8hue06NI/AAAAAAAAAfc/pPnpTtu3fwA/s1600/dependencies.png)

<o:p>&lt;/o:p&gt;

2.       **Setup JSF for your application

Dependency declaration is complete. The next step is to setup Faces for
your application.<o:p>&lt;/o:p&gt;

In your web.xml mention faces servlet
and the extensions you want to map with JSF and run your server. It should be
working.<o:p>&lt;/o:p&gt;

If not, then delete the web.xml and
retry. Here is one sample for ease.<o:p>&lt;/o:p&gt;

```
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xmlns="http://java.sun.com/xml/ns/javaee" xmlns:web="http://java.sun.com/xml/ns/javaee/web-app_2_5.xsd"
xsi:schemaLocation="http://java.sun.com/xml/ns/javaee
http://java.sun.com/xml/ns/javaee/web-app_2_5.xsd"
id="WebApp_ID" version="2.5">
<display-name>PrimeFaces Web Application&lt;/display-name&gt;
<!-- Change to "Production" when you are ready to deploy -->
<context-param>

javax.faces.PROJECT_STAGE&lt;/param-name&gt;

Development&lt;/param-value&gt;
&lt;/context-param&gt;
&lt;!-- Welcome page --&gt;
<welcome-file-list>
<welcome-file>faces/index.xhtml&lt;/welcome-file&gt;
&lt;/welcome-file-list&gt;
&lt;!-- JSF mapping --&gt;
<servlet>
<servlet-name>Faces Servlet&lt;/servlet-name&gt;
<servlet-class>javax.faces.webapp.FacesServlet&lt;/servlet-class&gt;
<load-on-startup>1&lt;/load-on-startup&gt;
&lt;/servlet&gt;
<!-- Map these files with JSF -->
<servlet-mapping>
<servlet-name>Faces Servlet&lt;/servlet-name&gt;
<url-pattern>/faces/*&lt;/url-pattern&gt;
&lt;/servlet-mapping&gt;
<servlet-mapping>
<servlet-name>Faces Servlet&lt;/servlet-name&gt;
<url-pattern>*.jsf&lt;/url-pattern&gt;
&lt;/servlet-mapping&gt;
<servlet-mapping>
<servlet-name>Faces Servlet&lt;/servlet-name&gt;
<url-pattern>*.faces&lt;/url-pattern&gt;
&lt;/servlet-mapping&gt;
<servlet-mapping>
<servlet-name>Faces Servlet&lt;/servlet-name&gt;
<url-pattern>*.xhtml&lt;/url-pattern&gt;
&lt;/servlet-mapping&gt;
&lt;/web-app&gt;

```

3.       **Add Spring MVC to **Faces

Now, add Spring MVC to your faces application.
You have to include  Spring context file
as context-param with param name contextConfigLocation and RequestContextListener
and ContextLoaderListener as your application listener.

So, basically you have to add three more tags in your web.xml.

Now, it’s time to write faces-config.xml. Here you mention SpringBeanFacesELResolver
as the EL Resolver for your faces application.

After adding these, restart the server to check if configuration is working.<o:p>&lt;/o:p&gt;

```
<context-param>

contextConfigLocation&lt;/param-name&gt;

/WEB-INF/resources/context.xml
&lt;/param-value&gt;
&lt;/context-param&gt;

org.springframework.web.context.ContextLoaderListener&lt;/listener-class&gt;
&lt;/listener&gt;

org.springframework.web.context.request.RequestContextListener
&lt;/listener-class&gt;
&lt;/listener&gt;

```

This is the faces-config file

```
<?xml version="1.0" encoding="utf-8"?>
<faces-config xmlns="http://java.sun.com/xml/ns/javaee"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://java.sun.com/xml/ns/javaee
http://java.sun.com/xml/ns/javaee/web-facesconfig_2_0.xsd"
version="2.0">
<application>
<el-resolver>org.springframework.web.jsf.el.SpringBeanFacesELResolver&lt;/el-resolver&gt;
&lt;/application&gt;
&lt;/faces-config&gt;

```

4.       **Add Hibernate to Spring

In your context file, DriverManagerDataSource
as data source, hibernate4.LocalSessionFactoryBean as sessionFactory and use
this in your DAO classes. Add POJO, hibernate configurations and mappings etc in the project.

Again restart the server and it is in complete configuration.<o:p>&lt;/o:p&gt;

```
&lt;/property&gt;&lt;/property&gt;&lt;/property&gt;&lt;/property&gt;&lt;/bean&gt;
&lt;/property&gt;&lt;/property&gt;&lt;/bean&gt;
&lt;/property&gt;&lt;/bean&gt;
&lt;/property&gt;&lt;/bean&gt;
&lt;/beans&gt;

```

Hibernate configuration

```
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE hibernate-configuration SYSTEM
"http://www.hibernate.org/dtd/hibernate-configuration-3.0.dtd">
<hibernate-configuration>
<session-factory>

org.hibernate.dialect.HSQLDialect
&lt;/property&gt;

org.hsqldb.jdbc.JDBCDriver
&lt;/property&gt;
<!-- Assume test is the database name -->

jdbc:hsqldb:hsql://localhost:1521/opt/db
&lt;/property&gt;

SA
&lt;/property&gt;

&lt;/property&gt;
<!-- List of XML mapping files -->
<mapping resource="resources/Employee.hbm.xml" />
&lt;/session-factory&gt;
</hibernate-configuration>

```

Mapping

```
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE hibernate-mapping PUBLIC
"-//Hibernate/Hibernate Mapping DTD//EN"
"http://www.hibernate.org/dtd/hibernate-mapping-3.0.dtd">
<hibernate-mapping>
<class name="bean.Employee" table="EMPLOYEE">
<meta attribute="class-description">
This class contains the employee detail.
&lt;/meta&gt;
<generator class="native"/>
&lt;/id&gt;

&lt;/class&gt;
&lt;/hibernate-mapping&gt;

```

**5.   Retrieve Spring beans in
Managed Beans
All
the setup is complete**. To retrieve Spring beans, in your Managed beans,
use @ManagedProperty on attributes.**

```
package managedbeans;
import javax.faces.bean.ManagedBean;
import javax.faces.bean.ManagedProperty;
import service.EmployeeService;
@ManagedBean(name = "editor")
public class EditorBean {
@ManagedProperty(name = "service", value = "#{service}")
private EmployeeService service;
private String value = "Editor";
/**
* @return the value
*/
public String getValue() {
return value;
}
/**
* @param value
*      the value to set
*/
public void setValue(String value) {
this.value = value;
}
/**
* @return the service
*/
public EmployeeService getService() {
return service;
}
/**
* @param service
*      the service to set
*/
public void setService(EmployeeService service) {
this.service = service;
}
public void save() {
System.out.println("Using service");
System.out.println(getValue());
System.out.println(service.getEmployees());
}
}

```

**
Enjoy the development with these components. They are really cool stuffs.

I've complete websites built on top of these three only. They really are helpful for faster development, better maintainability, less coding, good looking GUI. You are free to take a look in the below locations,

[https://assistant-palash90.rhcloud.com/req-resp.asst](https://assistant-palash90.rhcloud.com/req-resp.asst)

[https://temperatureconverter-palash90.rhcloud.com/](https://temperatureconverter-palash90.rhcloud.com/)

**That's it. If anything breaks, in between, you can post comment, I will check and revert back.

**If this helped you out, you can share this in your network as well. Sharing is caring...

**[Next](http://palash-90.blogspot.com/2014/12/configure-web-rest-and-web-service-from.html)

Palash Kanti Kundu

<div class='
