import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.AttributeType.MicroAggregationFunction;
import org.deidentifier.arx.AttributeType.Hierarchy.DefaultHierarchy;
// import org.deidentifier.arx.aggregates.HierarchyBuilder;
// import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
// import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased;
// import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Range;
// import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased.Order;
import org.deidentifier.arx.criteria.KAnonymity;

// import java.io.ObjectInputFilter.Config;
import java.nio.charset.StandardCharsets;


public class Tests {
    public static void main(String[] args) throws Exception{

        // String input_file = args[1];

        Data data = Data.create("adult_clean2.csv", StandardCharsets.UTF_8, ',');

		// File myobject = new File("hierarchy.txt");
        // Scanner myReader = new Scanner(myobject);

        // while (myReader.hasNextLine()) {
        //     String line = myReader.nextLine();
        //     String[] types = line.split(",");
            
		// 	if (types[1].equals("Insensitive")) {
		// 		data.getDefinition().setAttributeType(types[0], AttributeType.INSENSITIVE_ATTRIBUTE);
		// 	} else if (types[1].equals("Identifying")) {
		// 		data.getDefinition().setAttributeType(types[0], AttributeType.IDENTIFYING_ATTRIBUTE);
		// 	} else if (types[1].equals("Quasi_identifying")){
		// 		data.getDefinition().setAttributeType(types[0], AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		// 	} else if (types[1].equals("Sensitive")) {
		// 		data.getDefinition().setAttributeType(types[0], AttributeType.SENSITIVE_ATTRIBUTE);
		// 	} else {
		// 		data.getDefinition().setAttributeType(types[0], Hierarchy.create(types[1], StandardCharsets.UTF_8, ','));
		// 	}
        // }
        // myReader.close();
        
        // HierarchyBuilderIntervalBased<Double> builder1 = HierarchyBuilderIntervalBased.create(DataType.DECIMAL, new Range<Double>(0d,0d,0d), new Range<Double>(100d,100d,100d));
    
        // builder1.setAggregateFunction(DataType.DECIMAL.createAggregate().createIntervalFunction(true, false));
        // builder1.addInterval(0d, 10d);
        // builder1.addInterval(10d, 20d);
        // builder1.addInterval(20d, 30d);
        // builder1.addInterval(30d, 40d);
        // builder1.addInterval(40d, 50d);
        // builder1.addInterval(50d, 60d);
        // builder1.addInterval(60d, 70d);
        // builder1.addInterval(70d, 80d);
        // builder1.addInterval(80d, 90d);
        // builder1.addInterval(90d, 100d);
        // builder1.addInterval(100d, 200d);

        // HierarchyBuilderRedactionBased<?> builder2 = HierarchyBuilderRedactionBased.create(Order.RIGHT_TO_LEFT, Order.RIGHT_TO_LEFT,' ','*' );
        // builder2.setAlphabetSize(20263, 7);
        
        DefaultHierarchy hierarchy = Hierarchy.create();
        hierarchy.add("Male", "*");
        hierarchy.add("Female", "*");

        data.getDefinition().setDataType("Age", DataType.DECIMAL);
        data.getDefinition().setDataType("fnlwgt", DataType.DECIMAL);
        data.getDefinition().setDataType("capital_gain", DataType.DECIMAL);
        data.getDefinition().setDataType("capital_loss", DataType.DECIMAL);
        data.getDefinition().setDataType("hours_per_week", DataType.DECIMAL);
        data.getDefinition().setDataType("education_num", DataType.DECIMAL);
        // data.getDefinition().setDataType("workclass", DataType.ORDERED_STRING);

        data.getDefinition().setAttributeType("Age", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        // data.getDefinition().setAttributeType("education", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("fnlwgt", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("workclass", AttributeType.INSENSITIVE_ATTRIBUTE);
		// data.getDefinition().setAttributeType("education", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("education_num", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("marital_status", AttributeType.INSENSITIVE_ATTRIBUTE);
		data.getDefinition().setAttributeType("occupation", AttributeType.INSENSITIVE_ATTRIBUTE);
		data.getDefinition().setAttributeType("relationship", AttributeType.INSENSITIVE_ATTRIBUTE);
		data.getDefinition().setAttributeType("race", AttributeType.INSENSITIVE_ATTRIBUTE);
		data.getDefinition().setAttributeType("sex", hierarchy);
		data.getDefinition().setAttributeType("capital_gain", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("capital_loss", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("hours_per_week", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		data.getDefinition().setAttributeType("native_country", AttributeType.INSENSITIVE_ATTRIBUTE);
		data.getDefinition().setAttributeType("class", AttributeType.INSENSITIVE_ATTRIBUTE);

        data.getDefinition().setMicroAggregationFunction("Age", MicroAggregationFunction.createArithmeticMean());
        data.getDefinition().setMicroAggregationFunction("fnlwgt", MicroAggregationFunction.createArithmeticMean());
        data.getDefinition().setMicroAggregationFunction("hours_per_week", MicroAggregationFunction.createArithmeticMean());
        data.getDefinition().setMicroAggregationFunction("capital_gain", MicroAggregationFunction.createArithmeticMean());
        data.getDefinition().setMicroAggregationFunction("capital_loss", MicroAggregationFunction.createArithmeticMean());

        data.getDefinition().setMicroAggregationFunction("education_num", MicroAggregationFunction.createArithmeticMean());

        // data.getDefinition().setMicroAggregationFunction("workclass", MicroAggregationFunction.createMode());

        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration config = ARXConfiguration.create();

        config.addPrivacyModel(new KAnonymity(2));
        // System.out.println(config.getHeuristicSearchThreshold());
        config.setSuppressionLimit(0.2d);
        // config.setSuppressionAlwaysEnabled(false);

        ARXResult result = anonymizer.anonymize(data, config);
    
        result.getOutput(false).save("Test_aggregates.csv", ',');
    }

}
